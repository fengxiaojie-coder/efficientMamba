"""Quick PPG visualization for UBFC-Phys sessions.

Saves a summary figure containing:
1) Three sampled video frames (start/middle/end)
2) PPG waveform (using UBFC-Phys BVP as PPG ground truth)

Usage example:
	python -m src.utils.visualize_clips \
	  --src dataSet/UBFC-Phys \
	  --subject s2 \
	  --trial T1 \
	  --out results/ubfc_phys_preview
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def _resolve_subject_root(src: Path, subject: str) -> Path:
	"""Resolve UBFC-Phys subject path, including s1/s1 nested extraction layout."""
	subject_dir = src / subject
	if not subject_dir.exists():
		raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

	nested_dirs = [p for p in subject_dir.iterdir() if p.is_dir()]
	if len(nested_dirs) == 1 and nested_dirs[0].name.lower() == subject.lower():
		return nested_dirs[0]
	return subject_dir


def _read_video_preview(video_path: Path, sample_idx: list[int] | None = None) -> tuple[dict[int, np.ndarray], int, float]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if n_frames <= 0:
		cap.release()
		raise RuntimeError(f"Invalid frame count from video: {video_path}")

	if sample_idx is None:
		sample_idx = [0, n_frames // 2, max(0, n_frames - 1)]

	sampled_frames: dict[int, np.ndarray] = {}
	for idx in sample_idx:
		idx = int(np.clip(idx, 0, max(0, n_frames - 1)))
		cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		ok, frame = cap.read()
		if not ok:
			continue
		sampled_frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	cap.release()
	if not sampled_frames:
		raise RuntimeError(f"No preview frames decoded from video: {video_path}")
	return sampled_frames, n_frames, float(fps)


def _load_signal(csv_path: Path) -> np.ndarray:
	sig = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
	sig = np.asarray(sig, dtype=np.float32).squeeze()
	if sig.ndim == 0:
		sig = np.array([float(sig)], dtype=np.float32)
	return sig


def _session_paths(root: Path, subject: str, trial: str) -> tuple[Path, Path]:
	suffix = f"{subject}_{trial}"
	video_path = root / f"vid_{suffix}.avi"
	bvp_path = root / f"bvp_{suffix}.csv"
	return video_path, bvp_path


def visualize_subject_trial(src: Path, subject: str, trial: str, out_dir: Path) -> Path:
	root = _resolve_subject_root(src, subject)
	video_path, bvp_path = _session_paths(root, subject, trial)

	if not video_path.exists():
		raise FileNotFoundError(f"Missing session video: {video_path}")
	if not bvp_path.exists():
		raise FileNotFoundError(f"Missing BVP csv: {bvp_path}")

	# Sample only three frames to avoid loading large videos entirely into memory.
	preview_frames, n_frames, fps = _read_video_preview(video_path)
	ppg = _load_signal(bvp_path)

	sample_idx = [0, n_frames // 2, max(0, n_frames - 1)]

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{subject}_{trial}_overview.png"

	fig = plt.figure(figsize=(14, 6.5))
	gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2])

	for i, idx in enumerate(sample_idx):
		ax = fig.add_subplot(gs[0, i])
		frame = preview_frames.get(idx)
		if frame is None:
			ax.text(0.5, 0.5, f"Frame {idx}\nN/A", ha="center", va="center")
		else:
			ax.imshow(frame)
		ax.set_title(f"Frame {idx}")
		ax.axis("off")

	# Use normalized x-axis to avoid assuming exact sensor sampling rate alignment.
	x_ppg = np.linspace(0.0, 1.0, num=max(1, ppg.size), dtype=np.float32)
	x_frame_marks = [idx / float(max(1, n_frames - 1)) for idx in sample_idx]

	ax_ppg = fig.add_subplot(gs[1, :])
	ax_ppg.plot(x_ppg, ppg, linewidth=1.0, color="tab:red", label="PPG (from BVP)")
	for xm in x_frame_marks:
		ax_ppg.axvline(xm, linestyle="--", linewidth=0.8, alpha=0.5, color="gray")
	ax_ppg.set_title("PPG signal")
	ax_ppg.set_xlabel("Normalized timeline")
	ax_ppg.set_ylabel("Amplitude")
	ax_ppg.grid(True, linestyle="--", alpha=0.3)
	ax_ppg.legend(loc="upper right")

	duration_s = n_frames / float(max(1e-6, fps))
	fig.suptitle(
		f"UBFC-Phys preview | {subject} {trial} | frames={n_frames}, fps={fps:.2f}, duration={duration_s:.2f}s",
		fontsize=13,
	)
	fig.tight_layout()
	fig.savefig(out_path, dpi=140)
	plt.close(fig)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--src", type=str, default="dataSet/UBFC-Phys")
	parser.add_argument("--subject", type=str, default="s2")
	parser.add_argument("--trial", type=str, default="T1", choices=["T1", "T2", "T3"])
	parser.add_argument("--out", type=str, default="results/ubfc_phys_preview")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
	else:
		logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")

	out_path = visualize_subject_trial(
		src=Path(args.src),
		subject=args.subject,
		trial=args.trial,
		out_dir=Path(args.out),
	)
	logger.info("Saved UBFC-Phys preview figure to %s", out_path)


if __name__ == "__main__":
	main()
