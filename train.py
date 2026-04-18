"""
train.py
3-phase fine-tuning for EfficientNetB4, ViT-B/16, ResNet50.
Then tunes ensemble weights on validation set.
"""

import os, json, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset import make_loaders
from models import EfficientNetB4Classifier, ViTClassifier, ResNet50Classifier, EnsembleModel
from augmentation import get_train_transforms, get_val_transforms

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Project\Emseble\data\unified")
SAVE_DIR   = Path(r"C:\Project\Emseble\checkpoints")
BATCH_SIZE = 32
NUM_WORKERS= 4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASES = [
    dict(name="head_only",      epochs=5,  lr=1e-3),
    dict(name="partial_unfreeze", epochs=10, lr=1e-4),
    dict(name="full_finetune",  epochs=10, lr=1e-5),
]
PATIENCE = 5

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  val"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


def train_model(model, name, train_loader, val_loader, phases, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler()
    best_val_acc = 0
    best_path    = save_dir / f"{name}_best.pt"
    history      = []

    for phase_idx, phase in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"  {name}  |  Phase {phase_idx+1}: {phase['name']}  |  lr={phase['lr']}")
        print(f"{'='*60}")

        # Unfreeze strategy
        if phase["name"] == "head_only":
            model.freeze_backbone()
        elif phase["name"] == "partial_unfreeze":
            if hasattr(model, "unfreeze_last_blocks"):
                model.unfreeze_last_blocks()
        else:
            model.unfreeze_all()

        total, trainable = count_params(model)
        print(f"  Trainable params: {trainable:,} / {total:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase["lr"], weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase["epochs"])

        no_improve = 0
        for epoch in range(1, phase["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
            vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
            scheduler.step()

            print(f"  Ep {epoch:02d} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                  f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
                  f"{time.time()-t0:.1f}s")

            history.append(dict(phase=phase["name"], epoch=epoch,
                                tr_loss=tr_loss, tr_acc=tr_acc,
                                vl_loss=vl_loss, vl_acc=vl_acc))

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), best_path)
                no_improve = 0
                print(f"    ✓ New best val acc: {best_val_acc:.4f}")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stopping (patience={PATIENCE})")
                    break

    # Save history
    with open(save_dir / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val acc: {best_val_acc:.4f}  →  {best_path}")
    return best_path, best_val_acc


# ─── ENSEMBLE WEIGHT TUNING ──────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  collecting probs", leave=False):
        imgs = imgs.to(DEVICE)
        with autocast():
            logits = model(imgs)
        probs = torch.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def tune_ensemble_weights(eff_probs, vit_probs, res_probs, labels, n_classes):
    """Grid search over simplex weights."""
    best_acc, best_w = 0, (0.4, 0.4, 0.2)
    step = 0.1
    candidates = [(a, b, c)
                  for a in np.arange(0, 1.01, step)
                  for b in np.arange(0, 1.01, step)
                  for c in np.arange(0, 1.01, step)
                  if abs(a + b + c - 1.0) < 0.01]

    for w_e, w_v, w_r in tqdm(candidates, desc="  grid search"):
        probs = w_e * eff_probs + w_v * vit_probs + w_r * res_probs
        acc   = (probs.argmax(1) == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_w   = (w_e, w_v, w_r)

    print(f"  Best weights: effnet={best_w[0]:.1f}  vit={best_w[1]:.1f}  resnet={best_w[2]:.1f}")
    print(f"  Ensemble val acc: {best_acc:.4f}")
    return best_w


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=str(DATA_DIR))
    parser.add_argument("--save",  default=str(SAVE_DIR))
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    data_dir = Path(args.data)
    save_dir = Path(args.save)

    print(f"Device: {DEVICE}")
    print(f"Data:   {data_dir}")

    # ── Loaders
    train_loader, val_loader, test_loader, classes = make_loaders(
        data_dir, get_train_transforms(), get_val_transforms(),
        batch_size=args.batch, num_workers=NUM_WORKERS
    )
    n_classes = len(classes)
    print(f"Classes: {n_classes}")

    # Save class list
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    # ── Train each model
    models_cfg = [
        ("efficientnet_b4", EfficientNetB4Classifier(n_classes)),
        ("vit_b16",         ViTClassifier(n_classes)),
        ("resnet50",        ResNet50Classifier(n_classes)),
    ]

    best_paths = {}
    for name, model in models_cfg:
        model = model.to(DEVICE)
        best_path, best_acc = train_model(
            model, name, train_loader, val_loader, PHASES, save_dir)
        best_paths[name] = best_path

    # ── Load best weights
    eff_model = EfficientNetB4Classifier(n_classes).to(DEVICE)
    eff_model.load_state_dict(torch.load(best_paths["efficientnet_b4"]))

    vit_model = ViTClassifier(n_classes).to(DEVICE)
    vit_model.load_state_dict(torch.load(best_paths["vit_b16"]))

    res_model = ResNet50Classifier(n_classes).to(DEVICE)
    res_model.load_state_dict(torch.load(best_paths["resnet50"]))

    # ── Tune ensemble weights
    print("\nTuning ensemble weights on val set...")
    eff_probs, labels = collect_probs(eff_model, val_loader)
    vit_probs, _      = collect_probs(vit_model, val_loader)
    res_probs, _      = collect_probs(res_model, val_loader)

    best_w = tune_ensemble_weights(eff_probs, vit_probs, res_probs, labels, n_classes)

    # Save ensemble
    ensemble = EnsembleModel(eff_model, vit_model, res_model, weights=best_w)
    torch.save({
        "effnet":  eff_model.state_dict(),
        "vit":     vit_model.state_dict(),
        "resnet":  res_model.state_dict(),
        "weights": best_w,
        "classes": classes,
    }, save_dir / "ensemble.pt")

    print(f"\nEnsemble saved to {save_dir / 'ensemble.pt'}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
