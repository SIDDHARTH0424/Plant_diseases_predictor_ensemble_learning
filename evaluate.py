"""
evaluate.py
Full evaluation: TTA + MC Dropout uncertainty + confusion matrix + per-class report.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import make_loaders, PlantDiseaseDataset
from models import EfficientNetB4Classifier, ViTClassifier, ResNet50Classifier, EnsembleModel
from augmentation import get_val_transforms, get_tta_transforms

SAVE_DIR  = Path(r"C:\Project\Emseble\checkpoints")
DATA_DIR  = Path(r"C:\Project\Emseble\data\unified")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MC      = 10   # MC Dropout passes


def load_ensemble(save_dir, n_classes, weights):
    eff = EfficientNetB4Classifier(n_classes).to(DEVICE)
    vit = ViTClassifier(n_classes).to(DEVICE)
    res = ResNet50Classifier(n_classes).to(DEVICE)

    # Allow numpy scalar (required for PyTorch 2.6+)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    ckpt = torch.load(save_dir / "ensemble.pt", map_location=DEVICE, weights_only=False)
    eff.load_state_dict(ckpt["effnet"])
    vit.load_state_dict(ckpt["vit"])
    res.load_state_dict(ckpt["resnet"])
    w = ckpt["weights"]

    return EnsembleModel(eff, vit, res, weights=w)

@torch.no_grad()
def predict_tta(ensemble, dataset, tta_transforms):
    """Run TTA over test set: average predictions across augmentation variants."""
    all_preds, all_labels, all_conf, all_unc = [], [], [], []

    ensemble.eval()
    for img_path, label in tqdm(dataset.samples, desc="TTA eval"):
        import numpy as np
        from PIL import Image
        raw = np.array(Image.open(img_path).convert("RGB"))

        probs_list = []
        for tf in tta_transforms:
            inp = tf(image=raw)["image"].unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda'):
                p_mean, p_std = ensemble.predict_with_uncertainty(inp, n_passes=N_MC)
            probs_list.append(p_mean.cpu())

        avg_probs = torch.stack(probs_list).mean(0).squeeze(0)
        pred      = avg_probs.argmax().item()
        conf      = avg_probs.max().item()
        unc       = avg_probs.std().item()   # simple uncertainty proxy

        all_preds.append(pred)
        all_labels.append(label)
        all_conf.append(conf)
        all_unc.append(unc)

    return np.array(all_preds), np.array(all_labels), np.array(all_conf), np.array(all_unc)


def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(max(12, len(classes)), max(10, len(classes) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    with open(SAVE_DIR / "classes.json") as f:
        classes = json.load(f)
    n_classes = len(classes)

    ensemble = load_ensemble(SAVE_DIR, n_classes, None)

    test_ds = PlantDiseaseDataset(DATA_DIR, "test", transform=None)
    tta_tfs = get_tta_transforms()

    preds, labels, confs, uncs = predict_tta(ensemble, test_ds, tta_tfs)

    # ── Accuracy
    acc = (preds == labels).mean()
    print(f"\nTest accuracy (TTA): {acc:.4f}")

    # ── Confidence stats
    correct_conf   = confs[preds == labels].mean()
    incorrect_conf = confs[preds != labels].mean() if (preds != labels).any() else 0
    print(f"Mean confidence — correct: {correct_conf:.4f}  |  incorrect: {incorrect_conf:.4f}")

    # ── Low-confidence samples
    low_conf_mask = confs < 0.80
    print(f"Low-confidence predictions (<0.80): {low_conf_mask.sum()} / {len(preds)}")

    # ── Classification report
    report = classification_report(labels, preds, target_names=classes, digits=4)
    print("\n" + report)
    (SAVE_DIR / "classification_report.txt").write_text(report)

    # ── Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, classes, SAVE_DIR / "confusion_matrix.png")

    # ── Save predictions
    results = []
    for i, (p, l, c, u) in enumerate(zip(preds, labels, confs, uncs)):
        results.append(dict(
            true=classes[l], pred=classes[p],
            confidence=round(float(c), 4),
            uncertainty=round(float(u), 4),
            correct=bool(p == l)
        ))
    with open(SAVE_DIR / "test_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
