"""Frozen VGG-19 feature extractor used as the perceptual loss network.

Layer indices kept from ``torchvision.models.vgg19(...).features``:
    3  -> relu1_2
    8  -> relu2_2
    15 -> relu3_3  (content layer)
    24 -> relu4_3

The TransformNet emits images in [-1, 1] (tanh). VGG was trained on ImageNet
[0, 1]-ranged inputs normalized by the canonical mean/std. ``normalize`` is
responsible for mapping [-1, 1] -> [0, 1] and then applying the ImageNet
statistics in-module so callers do not need to worry about it.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
from torchvision import models

# Layer indices in vgg19.features
STYLE_LAYERS: List[int] = [3, 8, 15, 24]
CONTENT_LAYER: int = 15

VGGOutputs = namedtuple("VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


class VGGFeatures(nn.Module):
    """Frozen VGG-19 that returns the four feature maps used for style/content."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Only keep up to relu4_3 (index 24). 25 = inclusive of index 24.
        features = vgg.features[:25]

        # Freeze everything
        for p in features.parameters():
            p.requires_grad = False
        features.eval()

        self.features = features

        # ImageNet normalization constants registered as buffers so they move
        # with .to(device).
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] inputs to ImageNet-normalized tensors expected by VGG."""
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> VGGOutputs:
        x = self.normalize(x)
        outputs: List[torch.Tensor] = []
        want = set(STYLE_LAYERS)
        max_idx = max(STYLE_LAYERS)
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in want:
                outputs.append(x)
            if idx >= max_idx:
                break
        # outputs is in order of the indices traversed -> matches STYLE_LAYERS order
        return VGGOutputs(*outputs)


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute batched Gram matrices normalized by C*H*W.

    Args:
        features: tensor of shape (B, C, H, W).

    Returns:
        Tensor of shape (B, C, C) where each slice is the normalized Gram
        matrix of the corresponding feature map.
    """
    b, c, h, w = features.shape
    f = features.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)
