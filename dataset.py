"""
dataset.py
PyTorch Dataset for the unified plant disease dataset.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter


class PlantDiseaseDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform=None):
        """
        Args:
            root:      path to data/unified/
            split:     "train" | "val" | "test"
            transform: albumentations Compose transform
        """
        self.root      = Path(root) / split
        self.transform = transform
        self.samples   = []   # (path, label_idx)
        self.classes   = []
        self.class_to_idx = {}

        # Discover classes
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes     = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_class_weights(self):
        """Returns per-sample weights for WeightedRandomSampler."""
        counts = Counter(label for _, label in self.samples)
        total  = len(self.samples)
        weights = [total / counts[label] for _, label in self.samples]
        return torch.tensor(weights, dtype=torch.float)


def make_loaders(data_dir: str, train_tf, val_tf, batch_size=32, num_workers=4):
    train_ds = PlantDiseaseDataset(data_dir, "train", train_tf)
    val_ds   = PlantDiseaseDataset(data_dir, "val",   val_tf)
    test_ds  = PlantDiseaseDataset(data_dir, "test",  val_tf)

    # Balanced sampler for training
    sampler = WeightedRandomSampler(
        weights=train_ds.get_class_weights(),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
