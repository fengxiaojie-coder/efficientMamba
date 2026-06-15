"""Visualize UBFC-RPPG Dataset2 ground truth as scatter plots.

This script scans a UBFC-RPPG dataset root for ``ground_truth.txt`` files,
parses the three Dataset2 lines, and saves a 1x3 scatter-plot figure for each
subject:

1. PPG signal
2. Heart rate (HR)
3. Timestep (seconds)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET_ROOT = Path(r"E:\UclHomework\research project\DataSet\UBFC-rPPG")
DEFAULT_OUTPUT_DIR = Path("results") / "ubfc_ground_truth_scatter"


def read_ground_truth(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Read the three Dataset2 ground-truth lines from one file."""

	lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
	if len(lines) < 3:
		raise ValueError(f"{file_path} must contain at least 3 lines")

	ppg = np.fromstring(lines[0], sep=" ")
	heart_rate = np.fromstring(lines[1], sep=" ")
	timestep = np.fromstring(lines[2], sep=" ")

	if ppg.size == 0 or heart_rate.size == 0 or timestep.size == 0:
		raise ValueError(f"{file_path} contains empty ground-truth data")

	return ppg, heart_rate, timestep


def plot_ground_truth(file_path: Path, output_dir: Path) -> Path:
	"""Create a three-panel scatter plot for one ground_truth.txt file."""

	ppg, heart_rate, timestep = read_ground_truth(file_path)
	fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

	series = [
		(ppg, "PPG signal", "PPG value", "tab:blue"),
		(heart_rate, "Heart rate (HR)", "HR value", "tab:orange"),
		(timestep, "Timestep (seconds)", "Seconds", "tab:green"),
	]

	for ax, (values, title, ylabel, color) in zip(axes, series):
		x_axis = np.arange(values.size)
		ax.plot(x_axis, values, color=color, linewidth=1.0)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
		ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

	axes[-1].set_xlabel("Sample index")
	fig.suptitle(f"UBFC-RPPG ground truth: {file_path.parent.name}", fontsize=14)
	fig.tight_layout(rect=[0, 0.02, 1, 0.96])

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{file_path.parent.name}_ground_truth_scatter.png"
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	return output_path


def find_ground_truth_files(dataset_root: Path) -> list[Path]:
	"""Find all Dataset2 ground_truth.txt files under the dataset root."""

	return sorted(dataset_root.rglob("ground_truth.txt"))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot UBFC-RPPG ground truth scatter charts")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=DEFAULT_DATASET_ROOT,
		help="UBFC-RPPG dataset root directory",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory to save the scatter plot images",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	ground_truth_files = find_ground_truth_files(args.dataset_root)

	if not ground_truth_files:
		raise FileNotFoundError(
			f"No ground_truth.txt files found under {args.dataset_root}"
		)

	saved_paths = []
	for file_path in ground_truth_files:
		saved_paths.append(plot_ground_truth(file_path, args.output_dir))

	import logging
	logger = logging.getLogger(__name__)
	logger.info('Saved %d scatter plot figure(s) to %s', len(saved_paths), args.output_dir)


if __name__ == "__main__":
	main()
