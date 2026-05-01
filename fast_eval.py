import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset import make_loaders, PlantDiseaseDataset
from augmentation import get_val_transforms
from models import EfficientNetB4Classifier, ViTClassifier, ResNet50Classifier

SAVE_DIR  = Path(r"C:\Project\Emseble\checkpoints")
DATA_DIR  = Path(r"C:\Project\Emseble\data\unified")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(name, n_classes):
    if name == "EfficientNetB4":
        model = EfficientNetB4Classifier(n_classes).to(DEVICE)
        ckpt_path = SAVE_DIR / "efficientnet_b4_best.pt"
    elif name == "ViT-B/16":
        model = ViTClassifier(n_classes).to(DEVICE)
        ckpt_path = SAVE_DIR / "vit_b16_best.pt"
    elif name == "ResNet50":
        model = ResNet50Classifier(n_classes).to(DEVICE)
        ckpt_path = SAVE_DIR / "resnet50_best.pt"
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    return acc, precision, recall, f1

def main():
    with open(SAVE_DIR / "classes.json") as f:
        classes = json.load(f)
    n_classes = len(classes)

    val_tf = get_val_transforms()
    _, _, test_loader, _ = make_loaders(DATA_DIR, train_tf=val_tf, val_tf=val_tf, batch_size=32)

    results = []
    models_to_test = ["EfficientNetB4", "ViT-B/16", "ResNet50"]
    for m_name in models_to_test:
        print(f"\nLoading {m_name}...")
        model = load_model(m_name, n_classes)
        acc, prec, rec, f1 = evaluate_model(model, test_loader)
        results.append({
            "model": m_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
        print(f"{m_name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    print("\n\nFINAL RESULTS SUMMARY:")
    for r in results:
        print(f"{r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f}")

if __name__ == "__main__":
    main()
