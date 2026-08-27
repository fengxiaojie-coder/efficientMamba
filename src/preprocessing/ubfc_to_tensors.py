"""Convert UBFC dataset videos to clip tensors of shape [T,6,H,W].

Produces one .pt file per clip containing clip features, clip HR label,
and a frame-aligned PPG waveform segment for waveform-level evaluation.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from src.preprocess.ubfc_rPPG_dataset_info import read_ground_truth
from src.preprocessing.clip_builder.clip_builder_registry import register_default_preprocess_clip_builders
from src.preprocessing.normalization.normalization_registry import register_default_preprocess_normalizations
from src.preprocessing.pipelines.pipeline_registry import register_default_preprocess_entries
from src.preprocessing.registry import build_preprocess_pipeline, build_preprocess_preset
from src.preprocessing.roi.roi_registry import register_default_roi_strategies
from src.preprocessing.pipelines.ubfc_pipeline import PreprocessConfig, PreprocessPipeline

logger = logging.getLogger(__name__)


register_default_preprocess_entries()
register_default_preprocess_normalizations()
register_default_preprocess_clip_builders()
register_default_roi_strategies()

# Keep OpenCV single-threaded to reduce native decoder instability on shared servers.
try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def extract_frames(video_path: Path, img_size: int, decoder: str = 'auto') -> Tuple[np.ndarray, float]:
    # Prefer torchvision decoder first. It avoids some OpenCV/ffmpeg crashes seen on HPC nodes.
    if decoder in ('auto', 'torchvision'):
        try:
            import torchvision
            video, _, info = torchvision.io.read_video(str(video_path), pts_unit='sec')
            if video is not None and int(video.shape[0]) > 0:
                fps = float(info.get('video_fps', 30.0) or 30.0)
                # read_video returns RGB uint8 tensor [T, H, W, C]
                frames = video.numpy()
                return frames, fps
            if decoder == 'torchvision':
                raise RuntimeError(f'torchvision decoder returned empty video for: {video_path}')
        except Exception as e:
            if decoder == 'torchvision':
                raise RuntimeError(
                    f'torchvision decoder failed for {video_path}. '
                    'Install `av` (pip install av) and retry, or use --decoder auto/opencv.'
                ) from e
            logger.warning('torchvision video decode failed for %s, fallback to cv2: %s', video_path, e)

    if decoder == 'torchvision':
        raise RuntimeError(f'torchvision decoder failed for {video_path} and cv2 fallback is disabled')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # keep original frame size here; resizing / cropping handled later
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    frames = np.stack(frames, axis=0)  # (N,H,W,3)
    return frames, fps





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


def extract_clip_ppg(start_frame: int, clip_len: int, fps: float, ppg_times: np.ndarray, ppg_values: np.ndarray) -> np.ndarray:
    """Interpolate ground-truth PPG to clip frame timestamps.

    Returns an array of length clip_len aligned to frame times
    t = (start_frame + k) / fps, k=0..clip_len-1.
    """
    frame_times = (start_frame + np.arange(clip_len, dtype=np.float32)) / float(fps)
    if ppg_values.size == 0:
        return np.zeros((clip_len,), dtype=np.float32)
    if ppg_times.size != ppg_values.size or ppg_times.size == 0:
        # Fallback when timestamps are missing/inconsistent.
        src_x = np.linspace(0.0, 1.0, num=ppg_values.size, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num=clip_len, dtype=np.float32)
        return np.interp(dst_x, src_x, ppg_values.astype(np.float32)).astype(np.float32)

    # Ensure monotonic timestamps for interpolation.
    order = np.argsort(ppg_times)
    ts = ppg_times[order].astype(np.float32)
    sig = ppg_values[order].astype(np.float32)
    return np.interp(frame_times, ts, sig).astype(np.float32)



def _find_ubfc_phys_roots(subject_dir: Path) -> list[Path]:
    """Return candidate roots for UBFC-Phys files (handles s1/s1 nested extraction)."""
    roots = [subject_dir]
    nested = [p for p in subject_dir.iterdir() if p.is_dir()]
    if len(nested) == 1 and nested[0].name.lower() == subject_dir.name.lower():
        roots.append(nested[0])
    return roots


def _extract_clip_ppg_normalized(ppg_values: np.ndarray, start: int, clip_len: int, total_frames: int) -> np.ndarray:
    """Resample PPG to the clip frame window using normalized timeline [0, 1]."""
    if ppg_values.size == 0:
        return np.zeros((clip_len,), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=ppg_values.size, dtype=np.float32)
    denom = max(1, total_frames - 1)
    dst_x = (start + np.arange(clip_len, dtype=np.float32)) / float(denom)
    dst_x = np.clip(dst_x, 0.0, 1.0)
    return np.interp(dst_x, src_x, ppg_values.astype(np.float32)).astype(np.float32)


def _save_clip_payloads(
    clips: np.ndarray,
    out_dir: Path,
    clip_prefix: str,
    fps: float,
    roi: str,
    bbox,
    stride: int,
    clip_len: int,
    ppg: np.ndarray,
    hr: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> int:
    saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int((clips.shape[0] - 1) * stride + clip_len)
    for i in range(clips.shape[0]):
        start = i * stride
        if hr is not None and t is not None and hr.size > 0 and t.size > 0:
            hr_val = compute_clip_hr(start, clip_len, fps, t, hr)
            ppg_clip = extract_clip_ppg(start, clip_len, fps, t, ppg)
        else:
            hr_val = float(np.mean(ppg)) if ppg.size > 0 else 0.0
            ppg_clip = _extract_clip_ppg_normalized(ppg, start, clip_len, total_frames)

        tensor = torch.from_numpy(clips[i])
        fn = out_dir / f"{clip_prefix}_clip_{i:04d}.pt"
        payload = {
            'clip': tensor,
            'hr': float(hr_val),
            'ppg': ppg_clip,
            'fps': float(fps),
            'roi_type': roi,
            'roi_bbox': bbox,
        }
        torch.save(payload, fn)
        saved += 1
    return saved


def process_subject(subject_dir: Path, out_dir: Path, clip_len: int = 128, stride: int = 8, img_size: int = 72,
                    roi: str = 'bbox', roi_pad: float = 0.0, forehead_ratio: float = 0.20,
                    side_ratio: float = 0.20, bottom_ratio: float = 0.20,
                    phys_bvp_norm: str = 'zscore', decoder: str = 'auto',
                    align_nose_axis: bool = False,
                    face_mesh_first_frame_mask: bool = False,
                    center_face: bool = False,
                    canonical_face_mask: bool = False,
                    frame_independent_face_mesh: bool = False,
                    config: Optional[PreprocessConfig] = None):
    if config is None:
        config = PreprocessConfig(
            clip_len=clip_len,
            stride=stride,
            img_size=img_size,
            roi=roi,
            roi_pad=roi_pad,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
            phys_bvp_norm=phys_bvp_norm,
            decoder=decoder,
            align_nose_axis=align_nose_axis,
            face_mesh_first_frame_mask=face_mesh_first_frame_mask,
            center_face=center_face,
            canonical_face_mask=canonical_face_mask,
            frame_independent_face_mesh=frame_independent_face_mesh,
        )
    pipeline = PreprocessPipeline(config)
    return pipeline.process_subject(subject_dir, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', type=str, default='', help='Optional JSON file containing preprocessing pipeline configuration.')
    parser.add_argument('--preset', type=str, default='ubfc_default',
                        help='Registered preprocessing preset name from src.preprocessing.registry')
    parser.add_argument('--pipeline', type=str, default='ubfc_default',
                        help='Registered preprocessing pipeline name from src.preprocessing.registry')
    parser.add_argument('--clip_len', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--roi', type=str, default=None, choices=['full', 'haar', 'mediapipe', 'ellipse', 'bbox', 'face_mesh', 'forehead'],
                        help='ROI strategy: full frame, haar cascade, mediapipe bbox, ellipse, or compact bbox crop')
    parser.add_argument('--roi_pad', type=float, default=None, help='Extra padding ratio applied to detected bbox')
    parser.add_argument('--forehead_ratio', type=float, default=None,
                        help='Extra top expansion ratio for bbox ROI (relative to face-box height). Set 0 to disable.')
    parser.add_argument('--side_ratio', type=float, default=None,
                        help='Extra left/right expansion ratio for ROI (relative to face-box width).')
    parser.add_argument('--bottom_ratio', type=float, default=None,
                        help='Extra bottom expansion ratio for ROI (relative to face-box height).')
    parser.add_argument('--phys_bvp_norm', type=str, default=None, choices=['none', 'zscore'],
                        help='Normalization applied to saved PPG (both UBFC-rPPG and UBFC-Phys). Default: zscore.')
    parser.add_argument('--normalization_method', type=str, default=None, choices=['none', 'zscore'],
                        help='Registered waveform normalization method. Overrides --phys_bvp_norm when provided.')
    parser.add_argument('--clip_builder', type=str, default=None, choices=['raw_plus_diff', 'raw_only', 'diff_only', 'mean_centered'],
                        help='Registered clip construction method used after ROI extraction.')
    parser.add_argument('--decoder', type=str, default=None, choices=['auto', 'torchvision', 'opencv'],
                        help='Video decode backend. Use torchvision to avoid cv2 decode crashes on some servers.')
    parser.add_argument('--align_nose_axis', action='store_true',
                        help='For face_mesh ROI: rotate each frame so the nose bridge axis is vertical.')
    parser.add_argument('--face_mesh_first_frame_mask', action='store_true',
                        help='For face_mesh ROI: detect polygon from first frame only and reuse for all frames.')
    parser.add_argument('--center_face', action='store_true',
                        help='For face_mesh ROI: translate the nose center to the image center after alignment.')
    parser.add_argument('--canonical_face_mask', action='store_true',
                        help='For face_mesh ROI: enable align_nose_axis + center_face + face_mesh_first_frame_mask together.')
    parser.add_argument('--frame_independent_face_mesh', action='store_true',
                        help='For face_mesh ROI: run per-frame independent detection/alignment (no cross-frame shared detector state).')
    parser.add_argument('--subjects', type=str, default='',
                        help='Optional comma-separated subject folder names to include (e.g. s1,s2,subject9).')
    parser.add_argument('--skip_subjects', type=str, default='',
                        help='Optional comma-separated subject folder names to skip (e.g. s9,.ipynb_checkpoints).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logs')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    src = Path(args.src)
    out = Path(args.out)
    subjects = [p for p in src.iterdir() if p.is_dir() and not p.name.startswith('.')]

    include_set = {s.strip() for s in str(args.subjects).split(',') if s.strip()}
    if include_set:
        subjects = [p for p in subjects if p.name in include_set]

    skip_set = {s.strip() for s in str(args.skip_subjects).split(',') if s.strip()}
    if skip_set:
        subjects = [p for p in subjects if p.name not in skip_set]

    config = PreprocessConfig.from_json(args.config) if args.config else build_preprocess_preset(args.preset)
    if args.clip_len is not None:
        config.clip_len = args.clip_len
    if args.stride is not None:
        config.stride = args.stride
    if args.size is not None:
        config.img_size = args.size
    if args.roi is not None:
        config.roi = args.roi
    if args.roi_pad is not None:
        config.roi_pad = args.roi_pad
    if args.forehead_ratio is not None:
        config.forehead_ratio = args.forehead_ratio
    if args.side_ratio is not None:
        config.side_ratio = args.side_ratio
    if args.bottom_ratio is not None:
        config.bottom_ratio = args.bottom_ratio
    if args.phys_bvp_norm is not None:
        config.phys_bvp_norm = args.phys_bvp_norm
        config.normalization_method = args.phys_bvp_norm
    if args.normalization_method is not None:
        config.normalization_method = args.normalization_method
        config.phys_bvp_norm = args.normalization_method
    if args.clip_builder is not None:
        config.clip_builder = args.clip_builder
    if args.decoder is not None:
        config.decoder = args.decoder
    config.align_nose_axis = args.align_nose_axis
    config.face_mesh_first_frame_mask = args.face_mesh_first_frame_mask
    config.center_face = args.center_face
    config.canonical_face_mask = args.canonical_face_mask
    config.frame_independent_face_mesh = args.frame_independent_face_mesh

    logger.info('Found %d subject folders under %s', len(subjects), src)
    logger.info('Using preprocessing preset: %s', args.preset)
    logger.info('Using preprocessing config: %s', config.to_dict())

    pipeline = build_preprocess_pipeline(args.pipeline, config)
    if not hasattr(pipeline, 'process_subject'):
        raise TypeError(f'Pipeline {args.pipeline} does not implement process_subject(...)')

    total = 0
    iterator = tqdm(subjects, desc='Preprocessing UBFC', unit='subject') if tqdm is not None else subjects
    for s in iterator:
        n = pipeline.process_subject(s, out)
        logger.info("Processed %s: %d clips", s.name, n)
        total += n
        if tqdm is not None:
            iterator.set_postfix_str(f'latest={s.name}:{n}, total={total}')
    logger.info("Total clips saved: %d", total)


if __name__ == '__main__':
    main()
