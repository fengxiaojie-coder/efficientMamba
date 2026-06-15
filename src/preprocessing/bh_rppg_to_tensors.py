"""Convert BH-rPPG image sequences to clip tensors.

Each BH-rPPG sample is expected to have the form:

    <root>/<sample_name>/<sample_name>/Frame_XXXXX.png
    <root>/<sample_name>/sensor.csv
    <root>/<sample_name>/timestamps.csv
    <root>/<sample_name>/wave.csv

The script reads the second-level image directory, builds clip tensors of
shape [T, 6, H, W] (diff + raw RGB), and writes one .pt file per clip.
Each file stores a dictionary with at least:

    {'clip': Tensor[T,6,H,W], 'hr': float}

The output layout is organized by sample name so it stays readable even when
there are many clips per subject-like folder.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)


def _read_numeric_csv(path: Path) -> np.ndarray:
    """Read all numeric cells from a CSV file into a 1D float array."""

    values: list[float] = []
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    values.append(float(cell))
                except ValueError:
                    # Skip headers such as "SPO2", "PULSE", "Wave".
                    continue
    return np.asarray(values, dtype=np.float32)


def _read_sensor_pulse(sensor_csv: Path) -> np.ndarray:
    """Read the PULSE column from BH-rPPG sensor.csv."""

    pulses: list[float] = []
    with sensor_csv.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get('PULSE')
            if value is None:
                continue
            value = value.strip()
            if not value:
                continue
            try:
                pulses.append(float(value))
            except ValueError:
                continue
    return np.asarray(pulses, dtype=np.float32)


def _find_image_dir(sample_dir: Path) -> Path | None:
    """Find the second-level directory that contains Frame_*.png files."""

    direct = sample_dir / sample_dir.name
    if direct.is_dir() and any(direct.glob('Frame_*.png')):
        return direct

    for child in sorted(sample_dir.iterdir()):
        if child.is_dir() and any(child.glob('Frame_*.png')):
            return child
    return None


def _load_frames(image_dir: Path, img_size: int) -> np.ndarray:
    frame_files = sorted(image_dir.glob('Frame_*.png'))
    if not frame_files:
        raise RuntimeError(f'No frames found in {image_dir}')

    frames = []
    for frame_file in frame_files:
        with Image.open(frame_file) as img:
            frame = img.convert('RGB').resize((img_size, img_size), Image.Resampling.LANCZOS)
            frame = np.asarray(frame, dtype=np.uint8)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f'Failed to decode any frames in {image_dir}')

    return np.stack(frames, axis=0)


def _detect_face_bbox_haar(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(cascade_path)
    bboxes = []
    for i in range(min(30, frames.shape[0])):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        dets = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(dets) == 0:
            continue
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
            results = detector.process(img)
            if not results.detections:
                continue
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


def _apply_roi_and_resize(frames: np.ndarray, roi: str, pad: float, img_size: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if roi == 'full':
        resized = [cv2.resize(f, (img_size, img_size)) for f in frames]
        return np.stack(resized, axis=0), None

    bbox = None
    if roi == 'haar':
        bbox = _detect_face_bbox_haar(frames)
    elif roi == 'mediapipe':
        bbox = _detect_face_bbox_mediapipe(frames)

    if bbox is None:
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
    return np.stack(resized, axis=0), (int(x1), int(y1), int(x2), int(y2))


def _interpolate_series(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 1:
        return np.full(target_len, float(values[0]), dtype=np.float32)

    x_old = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def make_clips(frames: np.ndarray, clip_len: int, stride: int) -> np.ndarray:
    """Create [num_clips, T, 6, H, W] tensors using raw RGB and frame diffs."""

    n_frames, height, width, _ = frames.shape
    clips = []
    for start in range(0, max(1, n_frames - clip_len + 1), stride):
        window = frames[start:start + clip_len].astype(np.float32) / 255.0
        diffs = np.zeros_like(window)
        diffs[1:] = window[1:] - window[:-1]
        concat = np.concatenate([diffs, window], axis=-1)
        concat = np.transpose(concat, (0, 3, 1, 2))
        clips.append(concat)

    if not clips:
        return np.empty((0, clip_len, 6, height, width), dtype=np.float32)
    return np.stack(clips, axis=0)


def compute_clip_hr(hr_series: np.ndarray, start_frame: int, clip_len: int) -> float:
    end_frame = min(start_frame + clip_len, hr_series.size)
    if start_frame >= end_frame:
        return float(hr_series[-1])
    return float(np.mean(hr_series[start_frame:end_frame]))


def discover_sample_dirs(root_dir: Path) -> list[Path]:
    sample_dirs = []
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and not child.name.startswith('.'):
            if _find_image_dir(child) is not None:
                sample_dirs.append(child)
    return sample_dirs


def process_sample(sample_dir: Path, out_dir: Path, clip_len: int = 16, stride: int = 8, img_size: int = 72, max_clips: int = 0) -> int:
    image_dir = _find_image_dir(sample_dir)
    if image_dir is None:
        logger.info('Skipping %s: no Frame_*.png directory found', sample_dir)
        return 0

    sensor_csv = sample_dir / 'sensor.csv'
    timestamps_csv = sample_dir / 'timestamps.csv'
    wave_csv = sample_dir / 'wave.csv'
    if not sensor_csv.exists():
        logger.info('Skipping %s: missing sensor.csv', sample_dir)
        return 0

    frames = _load_frames(image_dir, img_size)
    pulse_values = _read_sensor_pulse(sensor_csv)
    if pulse_values.size == 0:
        logger.info('Skipping %s: no PULSE values in sensor.csv', sample_dir)
        return 0

    hr_series = _interpolate_series(pulse_values, frames.shape[0])
    # apply ROI cropping if requested via global variable (injected by CLI args)
    global _GLOBAL_ROI
    global _GLOBAL_ROI_PAD
    roi_bbox = None
    if '_GLOBAL_ROI' in globals() and _GLOBAL_ROI != 'full':
        frames, roi_bbox = _apply_roi_and_resize(frames, roi=_GLOBAL_ROI, pad=_GLOBAL_ROI_PAD, img_size=img_size)

    clips = make_clips(frames, clip_len=clip_len, stride=stride)
    if clips.shape[0] == 0:
        logger.info('Skipping %s: not enough frames for a clip', sample_dir)
        return 0

    sample_out_dir = out_dir / sample_dir.name
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = _read_numeric_csv(timestamps_csv) if timestamps_csv.exists() else np.asarray([], dtype=np.float32)
    wave = _read_numeric_csv(wave_csv) if wave_csv.exists() else np.asarray([], dtype=np.float32)

    saved = 0
    for i in range(clips.shape[0]):
        if max_clips > 0 and saved >= max_clips:
            break
        start = i * stride
        hr_val = compute_clip_hr(hr_series, start, clip_len)
        tensor = torch.from_numpy(clips[i])
        fn = sample_out_dir / f'{sample_dir.name}_clip_{i:04d}.pt'
        payload = {
            'clip': tensor,
            'hr': float(hr_val),
            'subject': sample_dir.name,
            'source_dir': str(sample_dir),
            'image_dir': str(image_dir),
            'frame_count': int(frames.shape[0]),
            'timestamps': torch.from_numpy(timestamps) if timestamps.size else None,
            'wave': torch.from_numpy(wave) if wave.size else None,
            'roi_type': globals().get('_GLOBAL_ROI', 'full'),
            'roi_bbox': roi_bbox,
        }
        torch.save(payload, fn)
        saved += 1

    logger.info('Processed %s: %d clips', sample_dir.name, saved)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert BH-rPPG image sequences to clip tensors')
    parser.add_argument('--src', required=True, help='BH-rPPG dataset root directory')
    parser.add_argument('--out', required=True, help='Output directory for .pt clips')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--max_subjects', type=int, default=0, help='Limit the number of sample folders processed')
    parser.add_argument('--max_clips_per_subject', type=int, default=0, help='Limit the number of clips written per sample folder')
    parser.add_argument('--roi', type=str, default='mediapipe', choices=['full', 'haar', 'mediapipe'],
                        help='ROI strategy to use when cropping frames prior to clip creation')
    parser.add_argument('--roi_pad', type=float, default=1.2, help='Padding multiplier for detected bbox')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    sample_dirs = discover_sample_dirs(src)
    if args.max_subjects and args.max_subjects > 0:
        sample_dirs = sample_dirs[: args.max_subjects]

    if not sample_dirs:
        raise SystemExit(f'No BH-rPPG sample directories found under {src}')

    # propagate ROI globals for process_sample convenience
    globals()['_GLOBAL_ROI'] = args.roi
    globals()['_GLOBAL_ROI_PAD'] = args.roi_pad

    total = 0
    for sample_dir in sample_dirs:
        n = process_sample(
            sample_dir,
            out,
            clip_len=args.clip_len,
            stride=args.stride,
            img_size=args.size,
            max_clips=args.max_clips_per_subject,
        )
        logger.info('Processed %s: %d clips', sample_dir.name, n)
        total += n

    logger.info('Total clips saved: %d', total)


if __name__ == '__main__':
    main()
