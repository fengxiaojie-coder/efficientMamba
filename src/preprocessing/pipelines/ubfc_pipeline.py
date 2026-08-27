from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.preprocess.ubfc_rPPG_dataset_info import read_ground_truth
from src.preprocessing.registry import (
    build_preprocess_clip_builder,
    build_preprocess_normalization,
    build_preprocess_stage,
)
from src.preprocessing.roi.roi_extract import transform_frames_with_roi


@dataclass
class PreprocessConfig:
    """Configuration for the modular preprocessing pipeline."""

    clip_len: int = 128
    stride: int = 8
    img_size: int = 72
    roi: str = 'bbox'
    roi_pad: float = 0.0
    forehead_ratio: float = 0.20
    side_ratio: float = 0.20
    bottom_ratio: float = 0.20
    phys_bvp_norm: str = 'zscore'
    normalization_method: str = 'zscore'
    clip_builder: str = 'raw_plus_diff'
    decoder: str = 'auto'
    align_nose_axis: bool = False
    face_mesh_first_frame_mask: bool = False
    center_face: bool = False
    canonical_face_mask: bool = False
    frame_independent_face_mesh: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> 'PreprocessConfig':
        with Path(path).open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


class ROIStage:
    """Applies ROI extraction and resizing to video frames."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def process(self, frames: np.ndarray) -> tuple[np.ndarray, Optional[object]]:
        return transform_frames_with_roi(
            frames,
            roi=self.config.roi,
            size=self.config.img_size,
            pad=self.config.roi_pad,
            forehead_ratio=self.config.forehead_ratio,
            side_ratio=self.config.side_ratio,
            bottom_ratio=self.config.bottom_ratio,
            align_nose_axis=self.config.align_nose_axis,
            face_mesh_first_frame_mask=self.config.face_mesh_first_frame_mask,
            center_face=self.config.center_face,
            canonical_face_mask=self.config.canonical_face_mask,
            frame_independent_face_mesh=self.config.frame_independent_face_mesh,
        )


class NormalizationStage:
    """Normalizes waveform targets such as BVP or PPG signals."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        method_name = config.normalization_method or config.phys_bvp_norm
        self.method = build_preprocess_normalization(method_name, config)

    def process(self, ppg: np.ndarray) -> np.ndarray:
        return self.method.process(ppg)


class SamplingStage:
    """Builds sliding-window clips from the preprocessed frame tensor."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.clip_builder = build_preprocess_clip_builder(config.clip_builder, config)

    def process(self, frames: np.ndarray) -> np.ndarray:
        return self.clip_builder.process(frames)


class PreprocessPipeline:
    """Config-driven preprocessing pipeline for UBFC-style rPPG datasets."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.roi_stage = build_preprocess_stage('roi', self.config)
        self.normalization_stage = build_preprocess_stage('normalization', self.config)
        self.sampling_stage = build_preprocess_stage('sampling', self.config)
        self.stages = {
            'roi_stage': self.roi_stage,
            'normalization_stage': self.normalization_stage,
            'sampling_stage': self.sampling_stage,
        }

    def normalize_ppg(self, ppg: np.ndarray) -> np.ndarray:
        return self.normalization_stage.process(ppg)

    def preprocess_frames(self, frames: np.ndarray) -> tuple[np.ndarray, Optional[object]]:
        return self.roi_stage.process(frames)

    def build_clips(self, frames: np.ndarray) -> np.ndarray:
        return self.sampling_stage.process(frames)

    def process_subject(self, subject_dir: Path, out_dir: Path) -> int:
        from src.preprocessing.ubfc_to_tensors import _save_clip_payloads, _find_ubfc_phys_roots, extract_frames

        gt_path = subject_dir / 'ground_truth.txt'
        video_path = None
        for ext in ('.avi',):
            candidate = subject_dir / f'vid{ext}'
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            vids = list(subject_dir.glob('*.avi'))
            if vids:
                video_path = vids[0]

        if gt_path.exists() and video_path is not None:
            ppg, hr, t = read_ground_truth(gt_path)
            ppg = self.normalize_ppg(ppg)
            frames, fps = extract_frames(video_path, self.config.img_size, decoder=self.config.decoder)
            frames, bbox = self.preprocess_frames(frames)
            clips = self.build_clips(frames)
            return _save_clip_payloads(
                clips=clips,
                out_dir=out_dir,
                clip_prefix=subject_dir.name,
                fps=fps,
                roi=self.config.roi,
                bbox=bbox,
                stride=self.config.stride,
                clip_len=self.config.clip_len,
                ppg=ppg,
                hr=hr,
                t=t,
            )

        phys_sessions: list[tuple[Path, Path, str]] = []
        for base in _find_ubfc_phys_roots(subject_dir):
            for vid in sorted(base.glob('vid_*.avi')):
                suffix = vid.stem[len('vid_'):]
                bvp = base / f'bvp_{suffix}.csv'
                if bvp.exists():
                    phys_sessions.append((vid, bvp, suffix))

        if not phys_sessions:
            return 0

        total_saved = 0
        for vid, bvp_path, suffix in phys_sessions:
            try:
                ppg = np.loadtxt(bvp_path, delimiter=',', dtype=np.float32)
                ppg = np.asarray(ppg, dtype=np.float32).squeeze()
                if ppg.ndim == 0:
                    ppg = np.array([float(ppg)], dtype=np.float32)
                ppg = self.normalize_ppg(ppg)
            except Exception:
                continue

            frames, fps = extract_frames(vid, self.config.img_size, decoder=self.config.decoder)
            frames, bbox = self.preprocess_frames(frames)
            clips = self.build_clips(frames)

            clip_prefix = f"{subject_dir.name}_{suffix}"
            total_saved += _save_clip_payloads(
                clips=clips,
                out_dir=out_dir,
                clip_prefix=clip_prefix,
                fps=fps,
                roi=self.config.roi,
                bbox=bbox,
                stride=self.config.stride,
                clip_len=self.config.clip_len,
                ppg=ppg,
                hr=None,
                t=None,
            )
        return total_saved


def make_clips(frames: np.ndarray, clip_len: int, stride: int) -> np.ndarray:
    N, H, W, C = frames.shape
    clips = []
    for start in range(0, max(1, N - clip_len + 1), stride):
        window = frames[start:start + clip_len].astype(np.float32) / 255.0
        diffs = np.zeros_like(window)
        diffs[1:] = window[1:] - window[:-1]
        concat = np.concatenate([diffs, window], axis=-1)
        concat = np.transpose(concat, (0, 3, 1, 2))
        clips.append(concat)
    if len(clips) == 0:
        return np.empty((0, clip_len, 6, H, W), dtype=np.float32)
    return np.stack(clips, axis=0)


__all__ = [
    'PreprocessConfig',
    'ROIStage',
    'NormalizationStage',
    'SamplingStage',
    'PreprocessPipeline',
    'make_clips',
]
