"""
inference.py
Single-image inference: OOD check → leaf detection → ensemble classify → cure lookup.
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast

from models import EfficientNetB4Classifier, ViTClassifier, ResNet50Classifier, EnsembleModel
from augmentation import get_tta_transforms

SAVE_DIR    = Path(r"C:\Project\Emseble\checkpoints")
KB_PATH     = Path(r"C:\Project\Emseble\knowledge_base.json")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.80
IMG_SIZE       = 224
PATCH_STRIDE   = IMG_SIZE // 2   # 50% overlap
N_MC           = 10


def load_ensemble():
    with open(SAVE_DIR / "classes.json") as f:
        classes = json.load(f)
    n = len(classes)

    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    ckpt = torch.load(SAVE_DIR / "ensemble.pt", map_location=DEVICE, weights_only=False)
    eff  = EfficientNetB4Classifier(n).to(DEVICE)
    vit  = ViTClassifier(n).to(DEVICE)
    res  = ResNet50Classifier(n).to(DEVICE)
    eff.load_state_dict(ckpt["effnet"])
    vit.load_state_dict(ckpt["vit"])
    res.load_state_dict(ckpt["resnet"])
    w    = ckpt["weights"]

    model = EnsembleModel(eff, vit, res, weights=w)
    model.eval()
    return model, classes


def load_knowledge_base():
    with open(KB_PATH) as f:
        return json.load(f)


# ─── OOD DETECTION ───────────────────────────────────────────────────────────
# Simple heuristic: if green channel dominance is very low, likely not a leaf.
def is_likely_plant(img_np: np.ndarray, threshold=0.08) -> bool:
    img = img_np.astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    # green dominance = fraction of pixels where G > R and G > B
    green_dominant = ((g > r) & (g > b)).mean()
    return green_dominant > threshold


# ─── YOLO LEAF DETECTION ─────────────────────────────────────────────────────
def detect_leaf_yolo(img_path: str):
    """
    Returns (cropped_img_np, was_detected).
    Falls back to full image if YOLO not available.
    """
    try:
        from ultralytics import YOLO
        yolo = YOLO(str(SAVE_DIR / "yolo_leaf.pt"))
        results = yolo(img_path, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Take highest-confidence box
            best = boxes[boxes.conf.argmax()]
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            img = np.array(Image.open(img_path).convert("RGB"))
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                return cropped, True
    except Exception:
        pass  # YOLO not trained yet — fall through
    # Fallback: return full image
    img = np.array(Image.open(img_path).convert("RGB"))
    return img, False


# ─── PATCH FALLBACK ──────────────────────────────────────────────────────────
def extract_patches(img_np: np.ndarray):
    """Extract overlapping 224×224 patches."""
    h, w = img_np.shape[:2]
    patches = []
    for y in range(0, h - IMG_SIZE + 1, PATCH_STRIDE):
        for x in range(0, w - IMG_SIZE + 1, PATCH_STRIDE):
            patches.append(img_np[y:y+IMG_SIZE, x:x+IMG_SIZE])
    if not patches:
        # Image smaller than patch — resize
        patches = [np.array(Image.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE)))]
    return patches


@torch.no_grad()
def classify_image(model, img_np: np.ndarray, tta_transforms, classes):
    """Run TTA + MC Dropout on a single image array."""
    probs_list = []
    for tf in tta_transforms:
        inp = tf(image=img_np)["image"].unsqueeze(0).to(DEVICE)
        with autocast():
            p_mean, _ = model.predict_with_uncertainty(inp, n_passes=N_MC)
        probs_list.append(p_mean.cpu())
    avg_probs = torch.stack(probs_list).mean(0).squeeze(0)
    return avg_probs


def classify_with_patch_fallback(model, img_np: np.ndarray, tta_transforms, classes):
    """Try full image first; fall back to patch voting if confidence too low."""
    # Resize to 224×224 for direct classification
    resized = np.array(Image.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE)))
    probs   = classify_image(model, resized, tta_transforms, classes)
    conf    = probs.max().item()

    if conf >= CONF_THRESHOLD:
        return probs, conf, False  # full leaf, no fallback

    # Patch fallback
    patches = extract_patches(img_np)
    patch_preds = []
    for patch in patches:
        p = classify_image(model, patch, tta_transforms, classes)
        patch_preds.append(p)
    avg_probs = torch.stack(patch_preds).mean(0)
    conf = avg_probs.max().item()
    return avg_probs, conf, True  # patch fallback used


# ─── SEVERITY ESTIMATION ────────────────────────────────────────────────────
def estimate_severity(probs, class_name: str) -> str:
    conf = probs.max().item()
    if "healthy" in class_name:
        return "None"
    if conf >= 0.90:
        return "High"
    elif conf >= 0.75:
        return "Moderate"
    else:
        return "Low"


# ─── MAIN PREDICT ────────────────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    """
    Full inference pipeline.
    Returns structured dict with disease info + cure recommendations.
    """
    model, classes = load_ensemble()
    kb             = load_knowledge_base()
    tta_transforms = get_tta_transforms()

    # 1. OOD check
    img_np = np.array(Image.open(image_path).convert("RGB"))
    if not is_likely_plant(img_np):
        return {
            "status":  "rejected",
            "message": "Image does not appear to contain a plant leaf. Please retake the photo.",
        }

    # 2. Leaf detection
    leaf_np, leaf_detected = detect_leaf_yolo(image_path)

    # 3. Classify
    probs, conf, patch_used = classify_with_patch_fallback(
        model, leaf_np, tta_transforms, classes)

    if patch_used and not leaf_detected:
        status_msg = "partial_leaf"
    else:
        status_msg = "ok"

    # 4. Top-3 predictions
    top3_idx  = probs.topk(3).indices.tolist()
    top3      = [(classes[i], round(probs[i].item(), 4)) for i in top3_idx]
    pred_class = top3[0][0]

    # 5. Low confidence warning
    low_conf_warning = conf < CONF_THRESHOLD

    # 6. Uncertainty
    unc = probs.std().item()

    # 7. Cure lookup
    cure = kb.get(pred_class, {
        "display_name":        pred_class.replace("_", " ").title(),
        "pathogen":            "Unknown",
        "immediate_action":    ["Consult local agricultural extension service"],
        "prevention":          [],
        "recommended_products": [],
        "organic_options":     [],
        "notes":               "Disease not in knowledge base yet.",
    })

    severity = estimate_severity(probs, pred_class)

    return {
        "status":          status_msg,
        "disease":         cure.get("display_name", pred_class),
        "class_id":        pred_class,
        "confidence":      round(conf, 4),
        "uncertainty":     round(unc, 4),
        "severity":        severity if not patch_used else f"{severity} (estimated — partial leaf)",
        "low_confidence_warning": low_conf_warning,
        "top_3":           top3,
        "pathogen":        cure.get("pathogen"),
        "immediate_action": cure.get("immediate_action", []),
        "prevention":      cure.get("prevention", []),
        "recommended_products": cure.get("recommended_products", []),
        "organic_options": cure.get("organic_options", []),
        "notes":           cure.get("notes", ""),
    }


if __name__ == "__main__":
    import sys, pprint
    path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"
    result = predict(path)
    pprint.pprint(result)
