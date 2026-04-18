"""
models.py
EfficientNetB4, ViT-B/16, ResNet50 — each with MC Dropout head.
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig
import timm


class MCDropout(nn.Module):
    """Always-active dropout for uncertainty estimation."""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=True)  # always on


# ─── EfficientNetB4 ───────────────────────────────────────────────────────────

class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True)
        in_features   = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            MCDropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, n=2):
        """Unfreeze last n blocks of EfficientNet."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.conv_head.parameters():
            p.requires_grad = True
        for p in self.backbone.bn2.parameters():
            p.requires_grad = True
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ─── ViT-B/16 ─────────────────────────────────────────────────────────────────

class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0)  # no head
        hidden_size = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            MCDropout(p=dropout_p),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)   # [B, hidden_size]
        return self.head(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, n=4):
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ─── ResNet50 ─────────────────────────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            MCDropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self):
        """Unfreeze layer3 + layer4 + fc."""
        for name, p in self.backbone.named_parameters():
            if any(x in name for x in ["layer3", "layer4", "fc"]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ─── Ensemble ────────────────────────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    Weighted soft-voting ensemble.
    weights: list of 3 floats summing to 1 (effnet, vit, resnet)
    """
    def __init__(self, effnet, vit, resnet, weights=(0.4, 0.4, 0.2)):
        super().__init__()
        self.effnet  = effnet
        self.vit     = vit
        self.resnet  = resnet
        w = torch.tensor(weights, dtype=torch.float)
        self.register_buffer("weights", w / w.sum())

    def forward(self, x):
        p_eff  = torch.softmax(self.effnet(x),  dim=-1)
        p_vit  = torch.softmax(self.vit(x),     dim=-1)
        p_res  = torch.softmax(self.resnet(x),  dim=-1)
        return (self.weights[0] * p_eff +
                self.weights[1] * p_vit +
                self.weights[2] * p_res)

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_passes=10):
        """MC Dropout: run n forward passes, return mean probs + std."""
        self.train()  # activate dropout
        preds = torch.stack([self.forward(x) for _ in range(n_passes)])
        self.eval()
        mean = preds.mean(0)
        std  = preds.std(0)
        return mean, std
