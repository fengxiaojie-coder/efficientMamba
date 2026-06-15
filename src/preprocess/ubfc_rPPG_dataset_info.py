"""Utilities for inspecting the UBFC-rPPG dataset layout and sample metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATASET_ROOT = Path("dataSet") / "UBFC-rPPG"


def read_ground_truth(ground_truth_path: Path):
    """Read Dataset2 ground truth and return parsed arrays."""

    lines = ground_truth_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{ground_truth_path} does not contain 3 lines")

    ppg = np.fromstring(lines[0], sep=" ")
    heart_rate = np.fromstring(lines[1], sep=" ")
    timestep = np.fromstring(lines[2], sep=" ")
    return ppg, heart_rate, timestep


def get_video_info(video_path: Path):
    """Return a lightweight video summary if OpenCV is available."""

    try:
        import cv2
    except Exception:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def find_subject_dirs(root_dir: Path = DATASET_ROOT):
    """Find UBFC subject folders under the dataset root."""

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")
    return sorted([path for path in root_dir.iterdir() if path.is_dir() and path.name.startswith("subject")])


def print_dataset_preview(preview_count: int = 3, root_dir: Path = DATASET_ROOT):
    """Print a short summary for the first few UBFC subjects."""

    subject_dirs = find_subject_dirs(root_dir)
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Dataset root: %s', root_dir)
    logger.info('Found %d subject folder(s)', len(subject_dirs))

    for subject_dir in subject_dirs[: min(preview_count, len(subject_dirs))]:
        video_path = subject_dir / "vid.avi"
        ground_truth_path = subject_dir / "ground_truth.txt"
        logger.info('\nSubject: %s', subject_dir.name)
        logger.info('  video: %s (%s)', video_path.name, 'exists' if video_path.exists() else 'missing')
        logger.info('  ground truth: %s (%s)', ground_truth_path.name, 'exists' if ground_truth_path.exists() else 'missing')

        if ground_truth_path.exists():
            ppg, heart_rate, timestep = read_ground_truth(ground_truth_path)
            logger.info('  ppg samples: %d, first 5: %s', ppg.size, np.round(ppg[:5], 4).tolist())
            logger.info('  hr samples: %d, first 5: %s', heart_rate.size, np.round(heart_rate[:5], 4).tolist())
            logger.info('  timestep samples: %d, first 5: %s', timestep.size, np.round(timestep[:5], 4).tolist())

        video_info = get_video_info(video_path)
        if video_info is None:
            if video_path.exists():
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
                logger.info('  video size: %.2f MB', file_size_mb)
            else:
                logger.info('  video info: unavailable')
        else:
            print(
                "  video info: "
                f"{video_info['frame_count']} frames, "
                f"{video_info['fps']:.2f} fps, "
                f"{video_info['width']}x{video_info['height']}"
            )