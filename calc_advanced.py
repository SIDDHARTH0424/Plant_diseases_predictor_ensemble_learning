import time
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataset import make_loaders
from augmentation import get_val_transforms
from models import EfficientNetB4Classifier, ViTClassifier, ResNet50Classifier, EnsembleModel

SAVE_DIR  = Path(r"C:\Project\Emseble\checkpoints")
DATA_DIR  = Path(r"C:\Project\Emseble\data\unified")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_size=(1, 3, 224, 224), n_iters=50):
    model.eval()
    dummy_input = torch.randn(input_size).to(DEVICE)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    return ((end - start) / n_iters) * 1000  # ms per inference

def load_ensemble(n_classes):
    eff = EfficientNetB4Classifier(n_classes).to(DEVICE)
    vit = ViTClassifier(n_classes).to(DEVICE)
    res = ResNet50Classifier(n_classes).to(DEVICE)
    
    # Allow numpy scalar (required for PyTorch 2.6+)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    ckpt = torch.load(SAVE_DIR / "ensemble.pt", map_location=DEVICE, weights_only=False)
    eff.load_state_dict(ckpt["effnet"])
    vit.load_state_dict(ckpt["vit"])
    res.load_state_dict(ckpt["resnet"])
    w = ckpt["weights"]

    model = EnsembleModel(eff, vit, res, weights=w)
    return model, eff, vit, res

def calc_top3_accuracy(model, dataloader):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Top-3 Eval"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            # top 3 indices
            _, top3_preds = outputs.topk(3, dim=1, largest=True, sorted=True)
            
            # check correctness
            for i in range(len(labels)):
                true_label = labels[i].item()
                preds = top3_preds[i].tolist()
                
                if true_label == preds[0]:
                    correct_top1 += 1
                if true_label in preds:
                    correct_top3 += 1
                total += 1
                
    return correct_top1/total, correct_top3/total

def main():
    with open(SAVE_DIR / "classes.json") as f:
        classes = json.load(f)
    n_classes = len(classes)

    print("Loading models...")
    ensemble, eff, vit, res = load_ensemble(n_classes)
    
    print("\nCalculating Model Parameters...")
    p_eff = count_parameters(eff)
    p_vit = count_parameters(vit)
    p_res = count_parameters(res)
    p_ens = p_eff + p_vit + p_res
    print(f"EfficientNetB4: {p_eff/1e6:.2f}M")
    print(f"ViT-B/16: {p_vit/1e6:.2f}M")
    print(f"ResNet50: {p_res/1e6:.2f}M")
    print(f"Ensemble Total: {p_ens/1e6:.2f}M")
    
    print("\nMeasuring Inference Latency (ms/image)...")
    l_eff = measure_latency(eff)
    l_vit = measure_latency(vit)
    l_res = measure_latency(res)
    l_ens = measure_latency(ensemble)
    print(f"EfficientNetB4: {l_eff:.2f} ms")
    print(f"ViT-B/16: {l_vit:.2f} ms")
    print(f"ResNet50: {l_res:.2f} ms")
    print(f"Ensemble Total: {l_ens:.2f} ms")
    
    print("\nCalculating Top-3 Accuracy on Test Set...")
    val_tf = get_val_transforms()
    _, _, test_loader, _ = make_loaders(DATA_DIR, train_tf=val_tf, val_tf=val_tf, batch_size=32)
    
    top1, top3 = calc_top3_accuracy(ensemble, test_loader)
    print(f"\nEnsemble Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Ensemble Top-3 Accuracy: {top3*100:.2f}%")

if __name__ == "__main__":
    main()
