"""Minimal PhysFormer baseline adapter for the EfficientMamba project.

This module reuses the PhysFormer code that you downloaded in the sibling
`PhysFormer/` folder and wraps it with the same input/output contract used by
the current training and evaluation code:

- input:  [B, T, C, H, W]
- output: [B, T, 1]

For compatibility with the existing preprocessed clips, the adapter consumes
the raw RGB channels from the 6-channel clip payload and upsamples spatially to
the PhysFormer default input size when needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _find_physformer_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'PhysFormer'
        if (candidate / 'model' / 'Physformer.py').is_file():
            return candidate
    return None


_PHYSFORMER_ROOT = _find_physformer_repo_root()
if _PHYSFORMER_ROOT is not None:
    physformer_root_str = str(_PHYSFORMER_ROOT)
    if physformer_root_str not in sys.path:
        sys.path.insert(0, physformer_root_str)

try:
    from model.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
except Exception as exc:  # pragma: no cover - optional local dependency
    ViT_ST_ST_Compact3_TDC_gra_sharp = None
    _PHYSFORMER_IMPORT_ERROR = exc
else:
    _PHYSFORMER_IMPORT_ERROR = None


class PhysFormerBaselineRegressor(nn.Module):
    """Minimal PhysFormer wrapper with the same call signature as the current model."""

    def __init__(
        self,
        frame_depth: int = 128,
        spatial_size: int = 128,
        theta: float = 0.7,
        gra_sharp: float = 2.0,
    ):
        super().__init__()
        if ViT_ST_ST_Compact3_TDC_gra_sharp is None:
            raise ImportError(
                'PhysFormer could not be imported from the sibling PhysFormer/ folder. '
                f'Last import error: {_PHYSFORMER_IMPORT_ERROR}'
            )

        self.frame_depth = int(frame_depth)
        self.spatial_size = int(spatial_size)
        self.gra_sharp = float(gra_sharp)
        self.backbone = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(self.frame_depth, self.spatial_size, self.spatial_size),
            patches=(4, 4, 4),
            dim=96,
            ff_dim=144,
            num_heads=4,
            num_layers=12,
            dropout_rate=0.1,
            theta=float(theta),
        )

    def _prepare_input(self, frames: Tensor) -> Tensor:
        if frames.dim() != 5:
            raise ValueError(f'Expected [B, T, C, H, W], got shape {tuple(frames.shape)}')

        if frames.size(1) != self.frame_depth:
            raise ValueError(
                f'PhysFormer baseline expects T=={self.frame_depth}, got T=={int(frames.size(1))}. '
                'Re-run preprocessing or pass a matching --frame_depth.'
            )

        if frames.size(2) == 6:
            # PhysFormer is a raw-RGB baseline; use the raw channels from the
            # existing EfficientPhys-style payload.
            frames = frames[:, :, 3:6, :, :]
        elif frames.size(2) != 3:
            raise ValueError(f'PhysFormer baseline expects 3 or 6 channels, got C=={int(frames.size(2))}')

        # Convert [B, T, C, H, W] -> [B, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()

        if frames.size(-1) != self.spatial_size or frames.size(-2) != self.spatial_size:
            b, c, t, h, w = frames.shape
            flat = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(self.spatial_size, self.spatial_size), mode='bilinear', align_corners=False)
            frames = flat.reshape(b, t, c, self.spatial_size, self.spatial_size).permute(0, 2, 1, 3, 4).contiguous()

        return frames

    def forward(self, frames: Tensor, roi_map: Tensor | None = None, return_attention: bool = False):
        x = self._prepare_input(frames)
        rppg, score1, score2, score3 = self.backbone(x, self.gra_sharp)
        output = rppg.unsqueeze(-1)

        if return_attention:
            # Keep the interface compatible with the current training loop.
            # The PhysFormer baseline does not expose the same attention-map
            # structure as EfficientPhys, so return None here.
            return output, None
        return output
