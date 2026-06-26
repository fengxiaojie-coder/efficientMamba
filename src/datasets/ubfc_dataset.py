"""Dataset wrapper for preprocessed clip .pt files.
Each .pt file is expected to contain {'clip': Tensor[T,6,H,W], 'hr': float, 'ppg': [T]}.
Returns (clip, ppg_waveform, roi_type, roi_bbox) where ppg_waveform is the ground-truth PPG sequence [T].
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class UBFCClipDataset(Dataset):
    def __init__(self, clips_dir: str | Path, augment: bool = False, frame_depth: int | None = None):
        self.clips_dir = Path(clips_dir)
        all_files = sorted(self.clips_dir.glob('**/*.pt'))
        if frame_depth is not None:
            # Keep only clips whose temporal dimension matches frame_depth.
            # We peek at just the metadata rather than loading the full tensor.
            filtered = []
            skipped = 0
            for f in all_files:
                try:
                    d = torch.load(f, weights_only=False)
                    if int(d['clip'].shape[0]) == frame_depth:
                        filtered.append(f)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
            if skipped:
                import logging
                logging.getLogger(__name__).warning(
                    'UBFCClipDataset: skipped %d clips whose T != %d (mixed dataset detected). '
                    'Re-run preprocessing with --clip_len %d to fix.',
                    skipped, frame_depth, frame_depth,
                )
            self.files: List[Path] = filtered
        else:
            self.files: List[Path] = all_files
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        data = torch.load(fn, weights_only=False)  # clip: (T,6,H,W), hr: float, ppg: [T]
        # clip: (T,6,H,W)
        clip = data['clip'].float()

        # Extract PPG waveform as primary target; fall back to HR if PPG missing
        ppg_data = data.get('ppg', None)
        if ppg_data is not None:
            ppg = torch.tensor(ppg_data, dtype=torch.float32).squeeze()  # [T]
        else:
            # Fallback: use HR scalar replicated to T frames (legacy behavior)
            hr_val = float(data.get('hr', 0.0))
            T = clip.shape[0]
            ppg = torch.full((T,), hr_val, dtype=torch.float32)

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

        return clip, ppg, roi_type, roi_bbox


if __name__ == '__main__':
    import logging
    logger = logging.getLogger(__name__)
    ds = UBFCClipDataset('data/ubfc_clips')
    logger.info('Found %d clips', len(ds))
    if len(ds) > 0:
        item = ds[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            c, p = item[0], item[1]
        else:
            c, p = item
        logger.info('clip shape %s ppg shape %s', c.shape, p.shape)
