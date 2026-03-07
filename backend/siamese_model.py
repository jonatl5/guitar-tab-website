"""Shared lightweight Siamese CNN components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Grayscale image shape used by training and inference: (height, width)
IMAGE_SIZE = (64, 128)


class SiameseCNN(nn.Module):
    """Small CNN that maps a tab crop to a normalized embedding."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        reduced_h = IMAGE_SIZE[0] // 8
        reduced_w = IMAGE_SIZE[1] // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_h * reduced_w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss with margin."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # label=1 means same class, label=0 means different classes.
        dist = torch.norm(emb_a - emb_b, p=2, dim=1)
        positive = label * torch.square(dist)
        negative = (1.0 - label) * torch.square(torch.clamp(self.margin - dist, min=0.0))
        return torch.mean(positive + negative), dist
