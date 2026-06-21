"""Evaluation utilities for EfficientMamba.

Usage:
    python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

from src.datasets.ubfc_dataset import UBFCClipDataset
from src.models.efficientphys_mamba import EfficientPhysMambaRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_dir', default='data/ubfc_clips')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frame_depth', type=int, default=64,
                        help='Temporal clip length expected by TSM front-end (should match preprocessing clip_len)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='If >0, limit number of clips evaluated (useful for smoke tests)')
    parser.add_argument('--examples', type=int, default=5,
                        help='Number of example waveform plots to save')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Default FPS to assume if clip metadata does not include fps')
    parser.add_argument('--max_subjects', type=int, default=0,
                        help='If >0, limit dataset to this many subjects (subject prefix before "_clip_")')
    parser.add_argument('--subjects', type=str, default='',
                        help='Comma-separated list of subject names to include (e.g. subject01,subject02)')
    parser.add_argument('--per_subject_limit', type=int, default=0,
                        help='If >0, limit number of clips per subject to this many')
    parser.add_argument('--split_mode', type=str, default='clip_random',
                        choices=['clip_random', 'subject_disjoint'],
                        help='How to select/aggregate clips: random by clip, or grouped by subject')
    args = parser.parse_args()

    # configure logging according to verbosity flag
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = UBFCClipDataset(args.clips_dir, frame_depth=args.frame_depth)

    # Subject-aware selection (mirror train.py behavior)
    def subject_from_name(fn: Path) -> str:
        name = fn.name
        return name.split('_clip_', 1)[0] if '_clip_' in name else name

    subj_map = {}
    for fn in ds.files:
        s = subject_from_name(fn)
        subj_map.setdefault(s, []).append(fn)

    if args.subjects:
        want = [s.strip() for s in args.subjects.split(',') if s.strip()]
        subj_map = {s: subj_map[s] for s in subj_map if s in want}

    subject_list = sorted(subj_map.keys())
    if args.max_subjects and args.max_subjects > 0:
        subject_list = subject_list[: args.max_subjects]

    filtered_files = []
    for s in subject_list:
        files = sorted(subj_map[s])
        if args.per_subject_limit and args.per_subject_limit > 0:
            files = files[: args.per_subject_limit]
        filtered_files.extend(files)

    if args.max_samples > 0 and args.split_mode == 'clip_random':
        filtered_files = filtered_files[: args.max_samples]

    ds.files = filtered_files

    model = EfficientPhysMambaRegressor(in_channels=6, frame_depth=args.frame_depth)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    import numpy as np

    # Helper: estimate HR (bpm) from a 1D waveform using FFT
    def estimate_hr_from_waveform(sig: np.ndarray, fps: float) -> float:
        sig = np.asarray(sig)
        if sig.size < 4:
            return float('nan')
        sig = sig - sig.mean()
        # apply a conservative band-pass via FFT to reduce harmonics/noise
        window = np.hamming(sig.size)
        sigw = sig * window
        n = sigw.size
        # frequency axis for rfft
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        # compute complex spectrum
        spec_complex = np.fft.rfft(sigw)

        # conservative physiological band (Hz) for filtering: 0.7 - 2.5 Hz (~42-150 bpm)
        bp_low, bp_high = 0.7, 2.5
        bp_mask = (freqs >= bp_low) & (freqs <= bp_high)
        if not bp_mask.any():
            return float('nan')

        # zero-out frequencies outside the band (simple FFT-domain bandpass)
        spec_filtered = spec_complex * bp_mask.astype(float)

        # reconstruct time-domain filtered signal (real)
        sig_filtered = np.fft.irfft(spec_filtered, n)

        # recompute spectrum of filtered signal and find peak within physiological band
        spec2 = np.abs(np.fft.rfft(sig_filtered * window))
        mask = bp_mask
        if not mask.any():
            return float('nan')
        idx = int(np.argmax(spec2[mask]))
        peak_freq = freqs[mask][idx]
        return float(peak_freq * 60.0)

    def extract_subject_name(file_name: str) -> str:
        if '_clip_' in file_name:
            return file_name.split('_clip_', 1)[0]
        return Path(file_name).stem

    preds = []
    fft_preds = []
    hrs = []
    fns = []
    records = []

    # Manual batching so we can access filenames / per-file fps
    total = len(ds.files)
    max_samples = int(args.max_samples) if args.max_samples and args.max_samples > 0 else total
    batch_size = int(args.batch_size)
    processed = 0
    with torch.no_grad():
        for i in range(0, max_samples, batch_size):
            if processed >= max_samples:
                break
            batch_files = ds.files[i:i + batch_size]
            clips = []
            batch_hrs = []
            batch_fps = []
            batch_roi_types = []
            batch_payloads = []
            for fn in batch_files:
                data = torch.load(fn, weights_only=False)  # clip: (T,6,H,W), hr: float
                batch_payloads.append(data)
                clips.append(data['clip'].float())
                batch_hrs.append(float(data.get('hr', 0.0)))
                batch_fps.append(float(data.get('fps', args.fps)))
                batch_roi_types.append(data.get('roi_type', 'full'))
            if len(clips) == 0:
                break
            clips = torch.stack(clips, dim=0)  # [B,T,6,H,W]
            clips_device = clips.to(device)

            # build roi_map if preprocessing saved roi_type metadata
            roi_map = None
            try:
                if any(rt is not None and rt != 'full' for rt in batch_roi_types):
                    B = clips_device.size(0)
                    H = clips_device.size(-2)
                    W = clips_device.size(-1)
                    mask = torch.ones((B, 1, H, W), device=device)
                    for i, rt in enumerate(batch_roi_types):
                        if isinstance(rt, bytes):
                            rt = rt.decode()
                        if rt is None or rt == 'full':
                            mask[i, 0] = 1.0
                        else:
                            ys = torch.linspace(-1, 1, H, device=device).unsqueeze(1).expand(H, W)
                            xs = torch.linspace(-1, 1, W, device=device).unsqueeze(0).expand(H, W)
                            dist2 = xs ** 2 + ys ** 2
                            g = torch.exp(-dist2 / (0.5 ** 2))
                            mask[i, 0] = g
                    roi_map = mask
            except Exception:
                roi_map = None

            # forward: get per-frame waveform by applying head to temporal features
            features = model.forward_features(clips_device, roi_map=roi_map)
            temporal = model.temporal(features)
            per_frame = model.head(temporal).squeeze(-1).cpu().numpy()  # [B,T]
            pooled_out = model(clips_device, roi_map=roi_map).cpu().numpy().ravel()

            # collect
            for j, fn in enumerate(batch_files):
                subject_name = extract_subject_name(fn.name)
                payload = batch_payloads[j]
                preds.append(float(pooled_out[j]))
                hrs.append(float(batch_hrs[j]))
                fns.append(fn.name)
                # FFT-based HR estimate from per-frame waveform
                fps_val = float(batch_fps[j]) if batch_fps[j] is not None else float(args.fps)
                sig = per_frame[j]
                fft_hr = estimate_hr_from_waveform(sig, fps=fps_val)
                fft_preds.append(float(fft_hr) if not np.isnan(fft_hr) else float('nan'))
                records.append({
                    'file': fn.name,
                    'subject': subject_name,
                    'signal': np.asarray(sig).copy(),
                    'true_ppg': np.asarray(payload.get('ppg')).copy() if payload.get('ppg') is not None else None,
                    'fps': fps_val,
                    'pred_hr': float(pooled_out[j]),
                    'fft_hr': float(fft_hr) if not np.isnan(fft_hr) else float('nan'),
                    'true_hr': float(batch_hrs[j]),
                })
                processed += 1
                if processed >= max_samples:
                    break

    preds = np.array(preds)
    fft_preds = np.array(fft_preds)
    hrs = np.array(hrs)
    # Compute metrics
    mae = float(np.nanmean(np.abs(preds - hrs)))
    mae_fft = float(np.nanmean(np.abs(fft_preds - hrs)))
    bias = float(np.nanmean(preds - hrs))
    bias_std = float(np.nanstd(preds - hrs))
    pred_std = float(np.nanstd(preds))
    pred_min = float(np.nanmin(preds)) if preds.size > 0 else float('nan')
    pred_max = float(np.nanmax(preds)) if preds.size > 0 else float('nan')
    logger.info('Model MAE: %.3f bpm, FFT MAE: %.3f bpm', mae, mae_fft)
    logger.info('Prediction spread: pred_std=%.6f, pred_min=%.3f, pred_max=%.3f, bias=%.3f±%.3f', pred_std, pred_min, pred_max, bias, bias_std)
    if pred_std < 1e-3:
        logger.warning('Predictions are nearly constant (pred_std=%.6f). This indicates potential mean-collapse.', pred_std)

    # Save CSV and plots
    out_dir = Path('eval_outputs') / 'ubfc'
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = out_dir / 'predictions_ubfc.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'pred_hr', 'fft_hr', 'true_hr'])
        for fn, p, fhr, t in zip(fns, preds.tolist(), fft_preds.tolist(), hrs.tolist()):
            writer.writerow([fn, p, fhr, t])

    # Save a simple subject-level summary CSV as well
    summary_csv = out_dir / 'subject_summary_ubfc.csv'
    from collections import defaultdict
    subj_stats = defaultdict(list)
    for file_name, pred, true in zip(fns, preds.tolist(), hrs.tolist()):
        subj = extract_subject_name(file_name)
        subj_stats[subj].append((pred, true))
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'n_clips', 'mean_pred_hr', 'mean_true_hr', 'mae'])
        for subj, vals in sorted(subj_stats.items()):
            preds_s = [v[0] for v in vals]
            trues_s = [v[1] for v in vals]
            n = len(vals)
            mean_pred = float(sum(preds_s) / n)
            mean_true = float(sum(trues_s) / n)
            mae_s = float(sum(abs(p - t) for p, t in zip(preds_s, trues_s)) / n)
            writer.writerow([subj, n, mean_pred, mean_true, mae_s])

    try:
        import matplotlib.pyplot as plt

        def save_ppg_prediction_plot(signal, fps, file_path, title):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8, 3))
            time_axis = np.arange(signal.size) / float(fps)
            plt.plot(time_axis, signal, label='predicted PPG', color='tab:blue')
            plt.xlabel('Time (s)')
            plt.ylabel('PPG amplitude')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()

        def save_full_prediction_figure(signal, fps, pred_hr, true_hr, fft_hr, file_name, sample_name):
            time_axis = np.arange(signal.size) / float(fps)
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

            axes[0].plot(time_axis, signal, color='tab:blue', linewidth=1.0)
            axes[0].set_title('Predicted PPG signal')
            axes[0].set_ylabel('PPG value')
            axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.35)

            axes[1].plot(time_axis, np.full_like(time_axis, pred_hr), color='tab:orange', linewidth=1.2, label='predicted HR')
            axes[1].axhline(true_hr, color='tab:red', linestyle='--', linewidth=1.0, label='true HR')
            if not np.isnan(fft_hr):
                axes[1].axhline(fft_hr, color='tab:green', linestyle=':', linewidth=1.0, label='fft HR')
            axes[1].set_title('Heart rate (HR)')
            axes[1].set_ylabel('HR value')
            axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            axes[1].legend(loc='best')

            axes[2].plot(time_axis, time_axis, color='tab:green', linewidth=1.0)
            axes[2].set_title('Timestep (seconds)')
            axes[2].set_xlabel('Sample index')
            axes[2].set_ylabel('Seconds')
            axes[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.35)

            fig.suptitle(f'Prediction overview: {sample_name}', fontsize=14)
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            (out_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / file_name, dpi=200, bbox_inches='tight')
            plt.close(fig)

        def save_subject_ppg_figure(subject_name, subject_records):
            subject_records = sorted(subject_records, key=lambda item: item['file'])
            fps_val = float(subject_records[0]['fps']) if subject_records else float(args.fps)
            signals = [np.asarray(item['signal']) for item in subject_records]
            combined_signal = np.concatenate(signals) if signals else np.asarray([])
            # combine true PPGs if available for overlay
            true_signals = [np.asarray(item['true_ppg']) for item in subject_records if item.get('true_ppg') is not None]
            combined_true = np.concatenate(true_signals) if true_signals else None
            time_axis = np.arange(combined_signal.size) / fps_val if combined_signal.size else np.asarray([])
            subject_dir = out_dir / subject_name
            subject_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(time_axis, combined_signal, color='tab:blue', linewidth=1.0, label='predicted PPG')

            if combined_true is not None and combined_true.size:
                # align lengths by truncation if needed
                minlen = min(combined_true.size, combined_signal.size)
                if minlen > 0:
                    ax.plot(time_axis[:minlen], combined_true[:minlen], color='tab:red', linestyle='--', linewidth=0.9, label='true PPG')

            boundary = 0
            for item, signal in zip(subject_records[:-1], signals[:-1]):
                boundary += signal.size
                ax.axvline(boundary / fps_val, color='gray', linestyle='--', linewidth=0.7, alpha=0.35)

            ax.set_title(f'Subject {subject_name} - combined predicted PPG')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PPG amplitude')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(subject_dir / f'{subject_name}_ppg_prediction.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        def save_subject_hr_figure(subject_name, subject_records):
            subject_records = sorted(subject_records, key=lambda item: item['file'])
            clip_index = np.arange(1, len(subject_records) + 1)
            pred_vals = np.array([item['pred_hr'] for item in subject_records], dtype=float)
            fft_vals = np.array([item['fft_hr'] for item in subject_records], dtype=float)
            true_vals = np.array([item['true_hr'] for item in subject_records], dtype=float)
            subject_dir = out_dir / subject_name
            subject_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(clip_index, pred_vals, marker='o', linewidth=1.2, label='predicted HR', color='tab:orange')
            ax.plot(clip_index, fft_vals, marker='o', linewidth=1.2, label='fft HR', color='tab:green')
            ax.plot(clip_index, true_vals, linestyle='--', linewidth=1.0, label='true HR', color='tab:red')
            ax.set_title(f'Subject {subject_name} - HR prediction by clip')
            ax.set_xlabel('Clip index')
            ax.set_ylabel('HR (bpm)')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(subject_dir / f'{subject_name}_hr_prediction.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Scatter: model and FFT estimates vs true HR
        plt.figure(figsize=(6, 6))
        plt.scatter(hrs, preds, alpha=0.6, s=6, label='model')
        plt.scatter(hrs, fft_preds, alpha=0.6, s=6, label='fft')
        mn = float(min(np.nanmin(hrs), np.nanmin(preds)))
        mx = float(max(np.nanmax(hrs), np.nanmax(preds)))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('True HR (bpm)')
        plt.ylabel('Estimated HR (bpm)')
        plt.title(f'Pred vs True (model MAE={mae:.2f}, fft MAE={mae_fft:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'pred_vs_true.png')

        # Save a few example waveforms (first N)
        ex_n = int(min(args.examples, len(fns)))
        for k in range(ex_n):
            data = torch.load(ds.files[k])
            clip = data['clip'].float().unsqueeze(0).to(device)
            # single-sample roi_map
            single_roi = data.get('roi_type', 'full')
            roi_map_single = None
            if single_roi is not None and single_roi != 'full':
                H = clip.size(-2)
                W = clip.size(-1)
                ys = torch.linspace(-1, 1, H, device=device).unsqueeze(1).expand(H, W)
                xs = torch.linspace(-1, 1, W, device=device).unsqueeze(0).expand(H, W)
                dist2 = xs ** 2 + ys ** 2
                g = torch.exp(-dist2 / (0.5 ** 2))
                roi_map_single = torch.tensor(g, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                feat = model.forward_features(clip, roi_map=roi_map_single)
                temp = model.temporal(feat)
                sig = model.head(temp).squeeze(-1).cpu().numpy()[0]
            fps_val = float(data.get('fps', args.fps))
            title = (
                f"{ds.files[k].name} | true={data.get('hr', 0):.1f} bpm | "
                f"model={preds[k]:.1f} bpm | fft={fft_preds[k]:.1f} bpm"
            )
            plt.figure(figsize=(8, 3))
            tvec = np.arange(sig.size) / fps_val
            plt.plot(tvec, sig, label='waveform', color='tab:orange')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(title)
            plt.tight_layout()
            subject_name = subject_from_name(ds.files[k])
            subject_dir = out_dir / subject_name
            subject_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(subject_dir / f'waveform_example_{k:02d}.png')
            plt.close()
            save_ppg_prediction_plot(
                sig,
                fps_val,
                subject_dir / f'ppg_prediction_example_{k:02d}.png',
                f'PPG prediction | {title}',
            )

            save_full_prediction_figure(
                signal=sig,
                fps=fps_val,
                pred_hr=float(preds[k]),
                true_hr=float(data.get('hr', 0.0)),
                fft_hr=float(fft_preds[k]),
                file_name=f'{subject_name}/full_prediction_example_{k:02d}.png',
                sample_name=ds.files[k].name,
            )

        subject_groups = {}
        for item in records:
            subject_groups.setdefault(item['subject'], []).append(item)

        for subject_name in sorted(subject_groups):
            save_subject_ppg_figure(subject_name, subject_groups[subject_name])
            save_subject_hr_figure(subject_name, subject_groups[subject_name])

        logger.info('Saved evaluation outputs to %s', out_dir)
    except Exception:
        import traceback
        logger.warning('Plot generation failed:')
        traceback.print_exc()
        logger.warning('matplotlib not available — skipping plots')


if __name__ == '__main__':
    main()
