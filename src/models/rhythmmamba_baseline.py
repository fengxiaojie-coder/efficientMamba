"""RhythmMamba baseline adapter for EfficientMamba train/eval pipelines.

This wrapper imports RhythmMamba from the sibling `RhythmMamba/` repository and
adapts it to this project's standard interface:

- input:  [B, T, C, H, W]
- output: [B, T, 1]
"""
from __future__ import annotations

import sys
from pathlib import Path

from torch import Tensor, nn
import torch.nn.functional as F


def _find_rhythmmamba_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for folder_name in ('RhythmMamba', 'rhythmmamba'):
            candidate = parent / folder_name
            if (candidate / 'neural_methods' / 'model' / 'RhythmMamba.py').is_file():
                return candidate
    return None


_RHYTHMMAMBA_ROOT = _find_rhythmmamba_repo_root()
if _RHYTHMMAMBA_ROOT is not None:
    rhythmmamba_root_str = str(_RHYTHMMAMBA_ROOT)
    if rhythmmamba_root_str not in sys.path:
        sys.path.insert(0, rhythmmamba_root_str)

try:
    from neural_methods.model.RhythmMamba import RhythmMamba
except Exception as exc:  # pragma: no cover - optional local dependency
    RhythmMamba = None
    _RHYTHMMAMBA_IMPORT_ERROR = exc
else:
    _RHYTHMMAMBA_IMPORT_ERROR = None


class RhythmMambaBaselineRegressor(nn.Module):
    """Minimal RhythmMamba wrapper with this project's call signature."""

    def __init__(
        self,
        frame_depth: int = 128,
        spatial_size: int = 72,
        in_channels: int = 6,
    ):
        super().__init__()
        if RhythmMamba is None:
            raise ImportError(
                'RhythmMamba could not be imported from sibling RhythmMamba/ folder. '
                f'Last import error: {_RHYTHMMAMBA_IMPORT_ERROR}'
            )

        self.frame_depth = int(frame_depth)
        self.spatial_size = int(spatial_size)
        self.in_channels = int(in_channels)
        # Keep upstream default configuration for a faithful baseline.
        self.backbone = RhythmMamba()

    def _prepare_input(self, frames: Tensor) -> Tensor:
        if frames.dim() != 5:
            raise ValueError(f'Expected [B, T, C, H, W], got shape {tuple(frames.shape)}')

        if frames.size(1) != self.frame_depth:
            raise ValueError(
                f'RhythmMamba baseline expects T=={self.frame_depth}, got T=={int(frames.size(1))}. '
                'Re-run preprocessing or pass a matching --frame_depth.'
            )

        # RhythmMamba expects raw RGB channels.
        if frames.size(2) == 6:
            frames = frames[:, :, 3:6, :, :]
        elif frames.size(2) != 3:
            raise ValueError(f'RhythmMamba baseline expects 3 or 6 channels, got C=={int(frames.size(2))}')

        if frames.size(-1) != self.spatial_size or frames.size(-2) != self.spatial_size:
            b, t, c, h, w = frames.shape
            flat = frames.reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(self.spatial_size, self.spatial_size), mode='bilinear', align_corners=False)
            frames = flat.reshape(b, t, c, self.spatial_size, self.spatial_size)

        return frames

    def forward(self, frames: Tensor, roi_map: Tensor | None = None, return_attention: bool = False):
        x = self._prepare_input(frames)
        rppg = self.backbone(x)  # upstream returns [B, T]

        if rppg.dim() == 3 and rppg.size(1) == 1:
            rppg = rppg.squeeze(1)
        elif rppg.dim() != 2:
            raise RuntimeError(f'Unexpected RhythmMamba output shape: {tuple(rppg.shape)}')

        if rppg.size(1) != self.frame_depth:
            rppg = F.interpolate(rppg.unsqueeze(1), size=self.frame_depth, mode='linear', align_corners=False).squeeze(1)

        output = rppg.unsqueeze(-1)
        if return_attention:
            return output, None
        return output
