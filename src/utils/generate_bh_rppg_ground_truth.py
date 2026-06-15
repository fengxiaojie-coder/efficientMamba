"""Generate BH-rPPG ground truth CSV and per-subject scatter plots in `results/`.

Reads `eval_outputs/bh_rppg/predictions_bh_rppg.csv` (produced by `src.eval`) and
writes:
- `results/bh_rppg_ground_truth.csv` : per-clip file/subject/true_hr/pred_hr/fft_hr
- `results/bh_rppg_ground_truth_scatter/subjectX_ground_truth_scatter.png` : per-subject scatter plot

Run:
    python src/utils/generate_bh_rppg_ground_truth.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def extract_subject_name(file_name: str) -> str:
    if '_clip_' in file_name:
        return file_name.split('_clip_', 1)[0]
    return Path(file_name).stem


def main():
    repo_root = Path(__file__).resolve().parents[2]
    pred_csv = repo_root / 'eval_outputs' / 'bh_rppg' / 'predictions_bh_rppg.csv'
        if not pred_csv.exists():
            logger.error('Predictions CSV not found at %s', pred_csv)
        sys.exit(1)

    rows = []
    with open(pred_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expected columns: file, pred_hr, fft_hr, true_hr
            rows.append(r)

    results_dir = repo_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Write consolidated ground truth CSV
    out_csv = results_dir / 'bh_rppg_ground_truth.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'subject', 'pred_hr', 'fft_hr', 'true_hr'])
        for r in rows:
            subj = extract_subject_name(r['file'])
            writer.writerow([r['file'], subj, r.get('pred_hr', ''), r.get('fft_hr', ''), r.get('true_hr', '')])

    # Group by subject and make per-subject scatter plots
    subj_map = defaultdict(list)
    for r in rows:
        subj = extract_subject_name(r['file'])
        try:
            pred = float(r.get('pred_hr', 'nan'))
        except Exception:
            pred = float('nan')
        try:
            fft = float(r.get('fft_hr', 'nan'))
        except Exception:
            fft = float('nan')
        try:
            true = float(r.get('true_hr', 'nan'))
        except Exception:
            true = float('nan')
        subj_map[subj].append({'pred': pred, 'fft': fft, 'true': true})

    scatter_dir = results_dir / 'bh_rppg_ground_truth_scatter'
    scatter_dir.mkdir(exist_ok=True)

    if plt is None:
            logger.warning('matplotlib not available; skipping plot generation.')
        return

    for subj, items in sorted(subj_map.items()):
        preds = np.array([it['pred'] for it in items], dtype=float)
        trues = np.array([it['true'] for it in items], dtype=float)
        fftv = np.array([it['fft'] for it in items], dtype=float)

        if preds.size == 0 or trues.size == 0:
            continue

        plt.figure(figsize=(6, 6))
        plt.scatter(trues, preds, alpha=0.7, s=10, label='model')
        plt.scatter(trues, fftv, alpha=0.6, s=8, label='fft')
        mn = float(min(np.nanmin(trues), np.nanmin(preds)))
        mx = float(max(np.nanmax(trues), np.nanmax(preds)))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('True HR (bpm)')
        plt.ylabel('Estimated HR (bpm)')
        plt.title(f'{subj} - BH-rPPG ground truth scatter')
        plt.legend()
        plt.tight_layout()
        out_png = scatter_dir / f'{subj}_ground_truth_scatter.png'
        plt.savefig(out_png, dpi=200)
        plt.close()

        logger.info('Wrote ground truth CSV to %s', out_csv)
        logger.info('Wrote per-subject scatter plots to %s', scatter_dir)


if __name__ == '__main__':
    main()
