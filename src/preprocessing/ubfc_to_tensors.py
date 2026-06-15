"""Convert UBFC dataset videos to clip tensors of shape [T,6,H,W].

Produces one .pt file per clip containing {'clip': Tensor[T,6,H,W], 'hr': float}.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from src.preprocess.ubfc_rPPG_dataset_info import read_ground_truth, get_video_info
import logging

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


def _detect_face_bbox_haar(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect face bbox across frames using OpenCV Haar cascade.

    Returns a single bbox (x1,y1,x2,y2) computed as the median of detected boxes,
    or None if no face found.
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(cascade_path)
    bboxes = []
    for i in range(min(30, frames.shape[0])):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        dets = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(dets) == 0:
            continue
        # pick largest
        areas = [w * h for (x, y, w, h) in dets]
        idx = int(np.argmax(areas))
        x, y, w, h = dets[idx]
        bboxes.append((x, y, x + w, y + h))
    if not bboxes:
        return None
    arr = np.array(bboxes, dtype=np.int32)
    x1 = int(np.median(arr[:, 0]))
    y1 = int(np.median(arr[:, 1]))
    x2 = int(np.median(arr[:, 2]))
    y2 = int(np.median(arr[:, 3]))
    return x1, y1, x2, y2


def _detect_face_bbox_mediapipe(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        import mediapipe as mp
    except Exception:
        return None
    mp_face = mp.solutions.face_detection
    bboxes = []
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        for i in range(min(30, frames.shape[0])):
            img = frames[i]
            # mediapipe expects RGB
            results = detector.process(img)
            if not results.detections:
                continue
            # pick first detection
            det = results.detections[0]
            h, w, _ = img.shape
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)
            bboxes.append((x1, y1, x2, y2))
    if not bboxes:
        return None
    arr = np.array(bboxes, dtype=np.int32)
    x1 = int(np.median(arr[:, 0]))
    y1 = int(np.median(arr[:, 1]))
    x2 = int(np.median(arr[:, 2]))
    y2 = int(np.median(arr[:, 3]))
    return x1, y1, x2, y2


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


def _apply_roi_and_resize(frames: np.ndarray, roi_type: str, pad: float, img_size: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    # frames: (N,H,W,3)
    if roi_type == 'full':
        resized = [cv2.resize(f, (img_size, img_size)) for f in frames]
        return np.stack(resized, axis=0), None

    bbox = None
    if roi_type == 'haar':
        bbox = _detect_face_bbox_haar(frames)
    elif roi_type == 'mediapipe':
        bbox = _detect_face_bbox_mediapipe(frames)

    if bbox is None:
        # fallback to full frame
        resized = [cv2.resize(f, (img_size, img_size)) for f in frames]
        return np.stack(resized, axis=0), None

    x1, y1, x2, y2 = bbox
    h, w = frames.shape[1], frames.shape[2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * pad
    bh = (y2 - y1) * pad
    x1n = int(max(0, cx - bw / 2.0))
    x2n = int(min(w, cx + bw / 2.0))
    y1n = int(max(0, cy - bh / 2.0))
    y2n = int(min(h, cy + bh / 2.0))
    cropped = [f[y1n:y2n, x1n:x2n] for f in frames]
    resized = [cv2.resize(f, (img_size, img_size)) for f in cropped]
    # return resized frames and the bbox used (in original frame coords)
    return np.stack(resized, axis=0), (int(x1), int(y1), int(x2), int(y2))


def process_subject(subject_dir: Path, out_dir: Path, clip_len: int = 16, stride: int = 8, img_size: int = 72, roi: str = 'mediapipe', roi_pad: float = 1.2):
    gt_path = subject_dir / 'ground_truth.txt'
    video_path = None
    for ext in ('.avi', '.mp4', '.mov'):
        candidate = subject_dir / f'vid{ext}'
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        # fallback: pick any video file
        vids = list(subject_dir.glob('*.avi')) + list(subject_dir.glob('*.mp4'))
        if vids:
            video_path = vids[0]
    if not gt_path.exists() or video_path is None:
        logger.info("Skipping %s, missing ground_truth or video", subject_dir)
        return 0

    ppg, hr, t = read_ground_truth(gt_path)
    frames, fps = extract_frames(video_path, img_size)
    frames, bbox = _apply_roi_and_resize(frames, roi_type=roi, pad=roi_pad, img_size=img_size)
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
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--roi', type=str, default='mediapipe', choices=['full', 'haar', 'mediapipe'],
                        help='ROI strategy: full frame, haar cascade, or mediapipe')
    parser.add_argument('--roi_pad', type=float, default=1.2, help='Padding multiplier applied to detected bbox')
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    subjects = [p for p in src.iterdir() if p.is_dir()]
    total = 0
    for s in subjects:
        n = process_subject(s, out, clip_len=args.clip_len, stride=args.stride, img_size=args.size,
                            roi=args.roi, roi_pad=args.roi_pad)
        logger.info("Processed %s: %d clips", s.name, n)
        total += n
    logger.info("Total clips saved: %d", total)


if __name__ == '__main__':
    main()
