"""Dataset wrapper for preprocessed clip .pt files.
Each .pt file is expected to contain {'clip': Tensor[T,6,H,W], 'hr': float}.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class UBFCClipDataset(Dataset):
    def __init__(self, clips_dir: str | Path, augment: bool = False):
        self.clips_dir = Path(clips_dir)
        self.files: List[Path] = sorted(self.clips_dir.glob('**/*.pt'))
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        data = torch.load(fn, weights_only=False)  # clip: (T,6,H,W), hr: float
        # clip: (T,6,H,W)
        clip = data['clip'].float()
        hr = torch.tensor(float(data.get('hr', 0.0)), dtype=torch.float32)
        # optional ROI metadata saved during preprocessing
        roi_type = data.get('roi_type', 'full')
        roi_bbox = data.get('roi_bbox', None)
        # ensure roi_bbox is always a 4-tuple of ints for consistent batching
        if roi_bbox is None:
            roi_bbox = (-1, -1, -1, -1)
        else:
            try:
                roi_bbox = tuple(int(x) for x in roi_bbox)
            except Exception:
                roi_bbox = (-1, -1, -1, -1)

        # simple on-the-fly augmentations for training
        if self.augment:
            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                clip = clip.flip(dims=[-1])
            # per-channel multiplicative jitter (small)
            c = clip.size(1)
            gains = torch.normal(mean=1.0, std=0.05, size=(c, 1, 1), dtype=clip.dtype, device=clip.device).clamp(0.8, 1.2)
            clip = clip * gains.unsqueeze(0)
            # additive gaussian noise
            noise = torch.randn_like(clip) * 0.01
            clip = clip + noise

        return clip, hr, roi_type, roi_bbox


if __name__ == '__main__':
    import logging
    logger = logging.getLogger(__name__)
    ds = UBFCClipDataset('data/ubfc_clips')
    logger.info('Found %d clips', len(ds))
    if len(ds) > 0:
        item = ds[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            c, h = item[0], item[1]
        else:
            c, h = item
        logger.info('clip shape %s hr %s', c.shape, h.item())
