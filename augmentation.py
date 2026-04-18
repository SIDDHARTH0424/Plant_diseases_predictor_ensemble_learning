"""
augmentation.py
Light augmentation pipeline for training — applied only to train split.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224


def get_train_transforms():
    return A.Compose([
        # Geometric
        A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        # Lighting / color
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomShadow(p=0.2),
        A.ToGray(p=0.05),
        # Camera artifacts
        A.ImageCompression(quality_range=(30, 70), p=0.3),
        A.GaussNoise(std_range=(10**0.5 / 255, 50**0.5 / 255), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # Occlusion / partial leaf
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, IMG_SIZE // 6),
            hole_width_range=(1, IMG_SIZE // 6),
            fill=0, p=0.3
        ),
        A.RandomCrop(height=int(IMG_SIZE * 0.85), width=int(IMG_SIZE * 0.85), p=0.2),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        # Normalize (ImageNet stats)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """Test-time augmentation variants."""
    base = [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.Rotate(p=1.0, limit=(-15, -15))] + base),
        A.Compose([A.Rotate(p=1.0, limit=(15, 15))] + base),
    ]