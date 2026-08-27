"""Minimal EfficientPhys baseline for the current waveform pipeline.

This wrapper keeps the EfficientPhys front-end already implemented in this
repository and removes the additional temporal backbone used by the
EfficientPhys-Mamba variant. The output contract matches the rest of the
project:

- input:  [B, T, C, H, W]
- output: [B, T, 1]
"""
from __future__ import annotations

from torch import Tensor, nn

from .efficientphys_front import EfficientPhysFront


class EfficientPhysBaselineRegressor(nn.Module):
    """EfficientPhys front-end + lightweight per-frame regression head."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 128, img_size: int = 72, frame_depth: int = 128):
        super().__init__()
        self.feature_extractor = EfficientPhysFront(
            in_channels=in_channels,
            nb_dense=embed_dim,
            img_size=img_size,
            frame_depth=frame_depth,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward_features(self, frames: Tensor, roi_map: Tensor | None = None, return_attention: bool = False):
        return self.feature_extractor.forward_features(frames, roi_map=roi_map, return_attention=return_attention)

    def forward(self, frames: Tensor, roi_map: Tensor | None = None, return_attention: bool = False):
        result = self.forward_features(frames, roi_map=roi_map, return_attention=return_attention)

        if return_attention:
            features, attn_dict = result
        else:
            features = result
            attn_dict = None

        bsz, steps, dim = features.shape
        flat = features.reshape(bsz * steps, dim)
        preds = self.head(flat).reshape(bsz, steps, 1)

        if return_attention:
            return preds, attn_dict
        return preds
