"""EfficientPhys + Mamba fusion prototype.

The goal of this module is to expose the part you asked for: an EfficientPhys-
style front-end that turns a video clip into a sequence of 1D embeddings, and a
temporal backbone that can be fed into Mamba.

If `mamba-ssm` is installed, the module uses it. Otherwise it falls back to a
GRU so the code remains runnable in a minimal environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None

# import the faithful EfficientPhys front-end if available in this package
from .efficientphys_front import EfficientPhysFront


class FrameEncoder(nn.Module):
    """Encode one RGB frame into a compact feature vector."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        return self.proj(x)


class EfficientPhysFeatureExtractor(nn.Module):
    """EfficientPhys-inspired raw + difference feature extractor.

    Input shape: [B, T, C, H, W]
    Output shape: [B, T, D]
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        # appearance / raw stream: we enable a small temporal-shift behaviour
        # (inspired by TSM) to let convolutional filters see temporal context
        self.raw_encoder = FrameEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.diff_encoder = FrameEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # how many channel groups to shift (higher -> stronger temporal mixing)
        self.temporal_shift_div = 8
    @staticmethod
    def _compute_frame_diff(frames: Tensor) -> Tensor:
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return diff

    def forward(self, frames: Tensor) -> Tensor:
        if frames.dim() != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got shape {tuple(frames.shape)}")

        batch_size, num_frames, channels, height, width = frames.shape
        # apply a light temporal shift to the raw appearance stream so each
        # frame encoder can access neighbouring-frame information without
        # using a full 3D conv. This is a cheap approximation of TSM.
        raw_frames = self._temporal_shift(frames, n_div=self.temporal_shift_div)
        raw_flat = raw_frames.reshape(batch_size * num_frames, channels, height, width)
        diff_flat = self._compute_frame_diff(frames).reshape(batch_size * num_frames, channels, height, width)

        raw_features = self.raw_encoder(raw_flat)
        diff_features = self.diff_encoder(diff_flat)
        fused = self.fuse(torch.cat([raw_features, diff_features], dim=-1))
        return fused.view(batch_size, num_frames, -1)

    @staticmethod
    def _temporal_shift(frames: Tensor, n_div: int = 8) -> Tensor:
        """Shift a small fraction of channels forward/backward across time.

        frames: [B, T, C, H, W]
        Returns a tensor the same shape with shifted channels.
        """
        if frames.dim() != 5:
            raise ValueError("_temporal_shift expects a 5D tensor [B,T,C,H,W]")

        B, T, C, H, W = frames.shape
        fold = max(1, C // n_div) if C >= n_div else 0
        if fold == 0:
            return frames

        out = frames.clone()
        # shift first fold channels forward (t -> t-1)
        if T > 1:
            out[:, :-1, 0:fold] = frames[:, 1:, 0:fold]
            # shift next fold channels backward (t -> t+1)
            end = min(2 * fold, C)
            out[:, 1:, fold:end] = frames[:, :-1, fold:end]
        return out


class TemporalBackbone(nn.Module):
    """GRU backbone with optional Mamba acceleration.

    Input/output shape: [B, T, D]
    """

    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4, use_mamba: bool = True):
        super().__init__()
        self.uses_mamba = bool(use_mamba and Mamba is not None)
        if self.uses_mamba:
            self.backbone = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        else:
            self.backbone = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if self.uses_mamba:
            x = self.backbone(x)
        else:
            x, _ = self.backbone(x)
        return self.norm(x)


class EfficientPhysMambaRegressor(nn.Module):
    """End-to-end fusion model for rPPG / HR regression."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 128, d_state: int = 16, d_conv: int = 4,
                 use_mamba: bool = True, img_size: int = 72, frame_depth: int = 128):
        super().__init__()
        # use the faithful EfficientPhys front to produce the d10 embedding
        # note: EfficientPhysFront accepts total per-frame channels (6 for
        # motion+appearance). It also supports 3-channel inputs by internal
        # duplication for compatibility.
        self.feature_extractor = EfficientPhysFront(
            in_channels=in_channels,
            nb_dense=embed_dim,
            img_size=img_size,
            frame_depth=frame_depth,
        )
        self.temporal = TemporalBackbone(d_model=embed_dim, d_state=d_state, d_conv=d_conv, use_mamba=use_mamba)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward_features(self, frames: Tensor, roi_map: Optional[Tensor] = None, return_attention: bool = False) -> Tensor | tuple:
        # delegate to EfficientPhysFront which returns [B, T, D] or ([B, T, D], attention_dict)
        return self.feature_extractor.forward_features(frames, roi_map=roi_map, return_attention=return_attention)

    def forward(self, frames: Tensor, roi_map: Optional[Tensor] = None, return_attention: bool = False) -> Tensor | tuple:
        result = self.forward_features(frames, roi_map=roi_map, return_attention=return_attention)

        if return_attention:
            features, attn_dict = result
        else:
            features = result
            attn_dict = None

        temporal_features = self.temporal(features)
        pooled = temporal_features.mean(dim=1)
        output = self.head(pooled)

        if return_attention:
            return output, attn_dict
        return output


@dataclass
class SmokeTestResult:
    output_shape: tuple[int, ...]
    feature_shape: tuple[int, ...]
    backbone: str


def smoke_test() -> SmokeTestResult:
    """Run a tiny forward pass on random data."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientPhysMambaRegressor(use_mamba=torch.cuda.is_available(), img_size=72, frame_depth=16).to(device)
    dummy = torch.randn(2, 16, 6, 72, 72, device=device)
    features = model.forward_features(dummy)
    output = model(dummy)
    backbone = "mamba" if model.temporal.uses_mamba else "gru"
    return SmokeTestResult(tuple(output.shape), tuple(features.shape), backbone)


if __name__ == "__main__":
    result = smoke_test()