"""Convert UBFC dataset videos to clip tensors of shape [T,6,H,W].

Produces one .pt file per clip containing {'clip': Tensor[T,6,H,W], 'hr': float}.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import logging
import cv2
import numpy as np
import torch

from src.preprocess.ubfc_rPPG_dataset_info import read_ground_truth, get_video_info
from src.preprocessing.roi_extract import transform_frames_with_roi

logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, img_size: int) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # keep original frame size here; resizing / cropping handled later
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames, axis=0)  # (N,H,W,3)
    return frames, fps



def make_clips(frames: np.ndarray, clip_len: int, stride: int) -> np.ndarray:
    # frames: (N,H,W,3) uint8/float
    N, H, W, C = frames.shape
    clips = []
    for start in range(0, max(1, N - clip_len + 1), stride):
        window = frames[start:start + clip_len].astype(np.float32) / 255.0
        # compute diffs
        diffs = np.zeros_like(window)
        diffs[1:] = window[1:] - window[:-1]
        # concat channel-wise: diff then raw -> shape (T,H,W,6)
        concat = np.concatenate([diffs, window], axis=-1)
        # reorder to (T,6,H,W)
        concat = np.transpose(concat, (0, 3, 1, 2))
        clips.append(concat)
    if len(clips) == 0:
        return np.empty((0, clip_len, 6, H, W), dtype=np.float32)
    return np.stack(clips, axis=0)


def compute_clip_hr(start_frame: int, clip_len: int, fps: float, hr_times: np.ndarray, hr_values: np.ndarray) -> float:
    start_t = start_frame / fps
    end_t = (start_frame + clip_len) / fps
    mask = (hr_times >= start_t) & (hr_times <= end_t)
    if mask.any():
        return float(np.mean(hr_values[mask]))
    # fallback: nearest timestamp
    idx = np.searchsorted(hr_times, (start_t + end_t) / 2.0)
    idx = np.clip(idx, 0, len(hr_values) - 1)
    return float(hr_values[idx])



def process_subject(subject_dir: Path, out_dir: Path, clip_len: int = 128, stride: int = 8, img_size: int = 72,
                    roi: str = 'bbox', roi_pad: float = 0.0, forehead_ratio: float = 0.20):
    gt_path = subject_dir / 'ground_truth.txt'
    video_path = None
    for ext in ('.avi'):
        candidate = subject_dir / f'vid{ext}'
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        # fallback: pick any video file
        vids = list(subject_dir.glob('*.avi'))
        if vids:
            video_path = vids[0]
    if not gt_path.exists() or video_path is None:
        logger.info("Skipping %s, missing ground_truth or video", subject_dir)
        return 0

    ppg, hr, t = read_ground_truth(gt_path)
    frames, fps = extract_frames(video_path, img_size)
    frames, bbox = transform_frames_with_roi(frames, roi=roi, size=img_size, pad=roi_pad, forehead_ratio=forehead_ratio)
    clips = make_clips(frames, clip_len=clip_len, stride=stride)
    saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(clips.shape[0]):
        start = i * stride
        hr_val = compute_clip_hr(start, clip_len, fps, t, hr)
        tensor = torch.from_numpy(clips[i])
        fn = out_dir / f"{subject_dir.name}_clip_{i:04d}.pt"
        payload = {'clip': tensor, 'hr': float(hr_val), 'roi_type': roi, 'roi_bbox': bbox}
        torch.save(payload, fn)
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--clip_len', type=int, default=128)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--roi', type=str, default='bbox', choices=['full', 'haar', 'mediapipe', 'ellipse', 'bbox'],
                        help='ROI strategy: full frame, haar cascade, mediapipe bbox, ellipse, or compact bbox crop')
    parser.add_argument('--roi_pad', type=float, default=0.0, help='Extra padding ratio applied to detected bbox')
    parser.add_argument('--forehead_ratio', type=float, default=0.20,
                        help='Extra top expansion ratio for bbox ROI (relative to face-box height). Set 0 to disable.')
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    subjects = [p for p in src.iterdir() if p.is_dir()]
    total = 0
    for s in subjects:
        n = process_subject(s, out, clip_len=args.clip_len, stride=args.stride, img_size=args.size,
                            roi=args.roi, roi_pad=args.roi_pad, forehead_ratio=args.forehead_ratio)
        logger.info("Processed %s: %d clips", s.name, n)
        total += n
    logger.info("Total clips saved: %d", total)


if __name__ == '__main__':
    main()
