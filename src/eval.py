"""Evaluation utilities for EfficientMamba.

Usage:
    python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt
    python -m src.eval --clips_dir data/ubfc_phys_clips_128 --checkpoint checkpoints/best.pt --dataset_name ubfc_phys
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
    parser.add_argument('--dataset_name', type=str, default='auto', choices=['auto', 'ubfc', 'ubfc_phys'],
                        help='Dataset tag used for eval output folders and filenames')
    args = parser.parse_args()

    # configure logging according to verbosity flag
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _infer_dataset_name(clips_dir: str, explicit: str) -> str:
        if explicit != 'auto':
            return explicit
        lower = str(clips_dir).lower()
        if 'phys' in lower:
            return 'ubfc_phys'
        return 'ubfc'

    dataset_name = _infer_dataset_name(args.clips_dir, args.dataset_name)
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
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # Log model configuration
    backbone_name = 'Mamba' if model.temporal.uses_mamba else 'GRU'
    total_params = sum(p.numel() for p in model.parameters())
    temporal_params = sum(p.numel() for p in model.temporal.parameters())
    logger.info('=' * 70)
    logger.info('Model Configuration')
    logger.info('=' * 70)
    logger.info('Architecture: EfficientPhysMambaRegressor')
    logger.info('Temporal Backbone: %s', backbone_name)
    logger.info('Frame depth: %d, Input channels: 6', args.frame_depth)
    logger.info('Total parameters: %s', f'{total_params:,}')
    logger.info('Temporal backbone parameters: %s (%.1f%%)', f'{temporal_params:,}', 100*temporal_params/total_params)
    logger.info('Device: %s', device)
    logger.info('Checkpoint: %s', args.checkpoint)
    logger.info('Dataset tag: %s', dataset_name)
    logger.info('=' * 70)

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

    def estimate_hr_from_peaks(sig: np.ndarray, fps: float) -> float:
        """Estimate HR (bpm) using peak detection after a light band-pass filter."""
        sig = np.asarray(sig, dtype=np.float32).reshape(-1)
        if sig.size < 8:
            return float('nan')
        sig = sig - float(np.mean(sig))
        try:
            from scipy.signal import butter, filtfilt, find_peaks
        except Exception:
            return float('nan')

        nyq = 0.5 * float(fps)
        if nyq <= 0:
            return float('nan')
        low = 0.75 / nyq
        high = 2.5 / nyq
        if not (0.0 < low < high < 1.0):
            return float('nan')

        try:
            b, a = butter(1, [low, high], btype='bandpass')
            filtered = filtfilt(b, a, sig)
            peaks, _ = find_peaks(filtered)
        except Exception:
            return float('nan')

        if peaks.size < 2:
            return float('nan')
        mean_interval = float(np.mean(np.diff(peaks))) / float(fps)
        if mean_interval <= 0.0:
            return float('nan')
        return float(60.0 / mean_interval)

    def _hr_metrics(pred_hr: np.ndarray, true_hr: np.ndarray) -> dict[str, float]:
        pred_hr = np.asarray(pred_hr, dtype=float)
        true_hr = np.asarray(true_hr, dtype=float)
        mask = np.isfinite(pred_hr) & np.isfinite(true_hr)
        if int(np.sum(mask)) == 0:
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'pearson_abs': float('nan'),
                'std_pred': float('nan'),
                'n': 0,
            }
        p = pred_hr[mask]
        t = true_hr[mask]
        pearson_abs = float('nan')
        if p.size >= 2 and float(np.std(p)) > 1e-8 and float(np.std(t)) > 1e-8:
            pearson_abs = float(abs(np.corrcoef(p, t)[0, 1]))
        return {
            'mae': float(np.mean(np.abs(p - t))),
            'rmse': float(np.sqrt(np.mean((p - t) ** 2))),
            'pearson_abs': pearson_abs,
            'std_pred': float(np.std(p)),
            'n': int(p.size),
        }

    def _zscore(sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float32)
        if sig.size == 0:
            return sig
        std = float(np.std(sig))
        if std < 1e-8:
            return np.zeros_like(sig)
        return (sig - float(np.mean(sig))) / std

    def _align_true_ppg_to_pred(pred_sig: np.ndarray, true_sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred_sig = np.asarray(pred_sig, dtype=np.float32)
        true_sig = np.asarray(true_sig, dtype=np.float32)
        if pred_sig.size == 0 or true_sig.size == 0:
            return pred_sig, true_sig
        if pred_sig.size == true_sig.size:
            return pred_sig, true_sig
        src_x = np.linspace(0.0, 1.0, num=true_sig.size, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num=pred_sig.size, dtype=np.float32)
        true_resampled = np.interp(dst_x, src_x, true_sig).astype(np.float32)
        return pred_sig, true_resampled

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or b.size < 3:
            return float('nan')
        a = _zscore(a)
        b = _zscore(b)
        if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    def _norm_rmse(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return float('nan')
        a = _zscore(a)
        b = _zscore(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def extract_subject_name(file_name: str) -> str:
        if '_clip_' in file_name:
            return file_name.split('_clip_', 1)[0]
        return Path(file_name).stem

    preds = []
    fft_preds = []
    fft_trues = []
    peak_preds = []
    peak_trues = []
    hrs = []
    fns = []
    records = []
    ppg_corrs = []
    ppg_nrmse = []

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
                pred_peak_hr = estimate_hr_from_peaks(sig, fps=fps_val)
                true_ppg = payload.get('ppg')
                if true_ppg is not None:
                    _pred_aligned, true_aligned = _align_true_ppg_to_pred(sig, np.asarray(true_ppg))
                    true_fft_hr = estimate_hr_from_waveform(true_aligned, fps=fps_val)
                    true_peak_hr = estimate_hr_from_peaks(true_aligned, fps=fps_val)
                else:
                    true_fft_hr = float(batch_hrs[j])
                    true_peak_hr = float(batch_hrs[j])

                fft_preds.append(float(fft_hr) if not np.isnan(fft_hr) else float('nan'))
                fft_trues.append(float(true_fft_hr) if not np.isnan(true_fft_hr) else float('nan'))
                peak_preds.append(float(pred_peak_hr) if not np.isnan(pred_peak_hr) else float('nan'))
                peak_trues.append(float(true_peak_hr) if not np.isnan(true_peak_hr) else float('nan'))
                records.append({
                    'file': fn.name,
                    'subject': subject_name,
                    'signal': np.asarray(sig).copy(),
                    'true_ppg': np.asarray(payload.get('ppg')).copy() if payload.get('ppg') is not None else None,
                    'fps': fps_val,
                    'pred_hr': float(pooled_out[j]),
                    'fft_hr': float(fft_hr) if not np.isnan(fft_hr) else float('nan'),
                    'fft_true_hr': float(true_fft_hr) if not np.isnan(true_fft_hr) else float('nan'),
                    'peak_hr': float(pred_peak_hr) if not np.isnan(pred_peak_hr) else float('nan'),
                    'peak_true_hr': float(true_peak_hr) if not np.isnan(true_peak_hr) else float('nan'),
                    'true_hr': float(batch_hrs[j]),
                })

                if true_ppg is not None:
                    pred_aligned, true_aligned = _align_true_ppg_to_pred(sig, np.asarray(true_ppg))
                    ppg_corrs.append(_safe_corr(pred_aligned, true_aligned))
                    ppg_nrmse.append(_norm_rmse(pred_aligned, true_aligned))

                processed += 1
                if processed >= max_samples:
                    break

    preds = np.array(preds)
    fft_preds = np.array(fft_preds)
    fft_trues = np.array(fft_trues)
    peak_preds = np.array(peak_preds)
    peak_trues = np.array(peak_trues)
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

    fft_metrics = _hr_metrics(fft_preds, fft_trues)
    peak_metrics = _hr_metrics(peak_preds, peak_trues)
    logger.info('----------------------------')
    logger.info('FFT Metric (EfficientPhys-style)')
    logger.info('Avg MAE across subjects: %.3f', fft_metrics['mae'])
    logger.info('Avg RMSE across subjects: %.3f', fft_metrics['rmse'])
    logger.info('Pearson FFT: %.4f', fft_metrics['pearson_abs'])
    logger.info('Std FFT: %.3f', fft_metrics['std_pred'])
    logger.info('----------------------------')
    logger.info('Peak Detection Metric (EfficientPhys-style)')
    logger.info('Avg MAE across subjects: %.3f', peak_metrics['mae'])
    logger.info('Avg RMSE across subjects: %.3f', peak_metrics['rmse'])
    logger.info('Pearson Peak: %.4f', peak_metrics['pearson_abs'])
    logger.info('Std Peak: %.3f', peak_metrics['std_pred'])
    logger.info('----------------------------')

    valid_corr = np.array([x for x in ppg_corrs if not np.isnan(x)], dtype=float)
    valid_nrmse = np.array([x for x in ppg_nrmse if not np.isnan(x)], dtype=float)
    if valid_corr.size > 0:
        logger.info(
            'PPG waveform metrics: mean_corr=%.4f, median_corr=%.4f, mean_nRMSE=%.4f (n=%d clips)',
            float(np.mean(valid_corr)),
            float(np.median(valid_corr)),
            float(np.mean(valid_nrmse)) if valid_nrmse.size > 0 else float('nan'),
            int(valid_corr.size),
        )
    else:
        logger.warning('No ground-truth PPG found in clip payloads, skipping waveform metrics.')

    # Save CSV and plots
    out_dir = Path('eval_outputs') / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = out_dir / f'predictions_{dataset_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'pred_hr', 'fft_hr', 'fft_true_hr', 'peak_hr', 'peak_true_hr', 'true_hr'])
        for fn, p, fhr, tfhr, phr, tphr, t in zip(
            fns,
            preds.tolist(),
            fft_preds.tolist(),
            fft_trues.tolist(),
            peak_preds.tolist(),
            peak_trues.tolist(),
            hrs.tolist(),
        ):
            writer.writerow([fn, p, fhr, tfhr, phr, tphr, t])

    hr_metric_csv = out_dir / f'hr_metrics_{dataset_name}.csv'
    with open(hr_metric_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'mae', 'rmse', 'pearson_abs', 'std_pred', 'n'])
        writer.writerow(['fft', fft_metrics['mae'], fft_metrics['rmse'], fft_metrics['pearson_abs'], fft_metrics['std_pred'], fft_metrics['n']])
        writer.writerow(['peak', peak_metrics['mae'], peak_metrics['rmse'], peak_metrics['pearson_abs'], peak_metrics['std_pred'], peak_metrics['n']])

    summary_txt = out_dir / f'hr_metrics_summary_{dataset_name}.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('----------------------------\n')
        f.write('FFT Metric\n')
        f.write(f"Avg MAE across subjects: {fft_metrics['mae']}\n")
        f.write(f"Avg RMSE across subjects: {fft_metrics['rmse']}\n")
        f.write(f"Pearson FFT: {fft_metrics['pearson_abs']}\n")
        f.write(f"Std FFT: {fft_metrics['std_pred']}\n")
        f.write('----------------------------\n')
        f.write('Peak Detection Metric\n')
        f.write(f"Avg MAE across subjects: {peak_metrics['mae']}\n")
        f.write(f"Avg RMSE across subjects: {peak_metrics['rmse']}\n")
        f.write(f"Pearson Peak: {peak_metrics['pearson_abs']}\n")
        f.write(f"Std Peak: {peak_metrics['std_pred']}\n")
        f.write('----------------------------\n')

    ppg_csv = out_dir / f'ppg_metrics_{dataset_name}.csv'
    with open(ppg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'subject', 'ppg_corr', 'ppg_nrmse'])
        for rec in records:
            true_ppg = rec.get('true_ppg')
            corr = float('nan')
            nrmse = float('nan')
            if true_ppg is not None:
                pred_aligned, true_aligned = _align_true_ppg_to_pred(np.asarray(rec['signal']), np.asarray(true_ppg))
                corr = _safe_corr(pred_aligned, true_aligned)
                nrmse = _norm_rmse(pred_aligned, true_aligned)
            writer.writerow([rec['file'], rec['subject'], corr, nrmse])

    # Save a simple subject-level summary CSV as well
    summary_csv = out_dir / f'subject_summary_{dataset_name}.csv'
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

    ppg_subject_csv = out_dir / f'subject_ppg_summary_{dataset_name}.csv'
    with open(ppg_subject_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'n_clips_with_ppg', 'mean_ppg_corr', 'median_ppg_corr', 'mean_ppg_nrmse'])
        for subj, items in sorted(subj_stats.items()):
            subj_records = [r for r in records if r['subject'] == subj and r.get('true_ppg') is not None]
            corr_vals = []
            nrmse_vals = []
            for rec in subj_records:
                pred_aligned, true_aligned = _align_true_ppg_to_pred(np.asarray(rec['signal']), np.asarray(rec['true_ppg']))
                c = _safe_corr(pred_aligned, true_aligned)
                e = _norm_rmse(pred_aligned, true_aligned)
                if not np.isnan(c):
                    corr_vals.append(c)
                if not np.isnan(e):
                    nrmse_vals.append(e)
            n = len(corr_vals)
            writer.writerow([
                subj,
                n,
                float(np.mean(corr_vals)) if n > 0 else float('nan'),
                float(np.median(corr_vals)) if n > 0 else float('nan'),
                float(np.mean(nrmse_vals)) if len(nrmse_vals) > 0 else float('nan'),
            ])

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

        def save_ppg_comparison_plot(pred_signal, true_signal, fps, file_path, title):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pred_signal, true_signal = _align_true_ppg_to_pred(np.asarray(pred_signal), np.asarray(true_signal))
            pred_z = _zscore(pred_signal)
            true_z = _zscore(true_signal)
            time_axis = np.arange(pred_z.size) / float(fps)
            corr = _safe_corr(pred_z, true_z)
            nrmse = _norm_rmse(pred_z, true_z)

            plt.figure(figsize=(10, 3.5))
            plt.plot(time_axis, true_z, label='true PPG (z-score)', color='tab:red', linestyle='--', linewidth=1.0)
            plt.plot(time_axis, pred_z, label='predicted PPG (z-score)', color='tab:blue', linewidth=1.1)
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized amplitude')
            plt.title(f'{title} | corr={corr:.3f} nRMSE={nrmse:.3f}')
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(file_path, dpi=220, bbox_inches='tight')
            plt.close()

        def save_ppg_comparison_plot_5s(pred_signal, true_signal, fps, file_path, title, window_seconds: float = 5.0):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pred_signal, true_signal = _align_true_ppg_to_pred(np.asarray(pred_signal), np.asarray(true_signal))
            if pred_signal.size == 0 or true_signal.size == 0:
                logger.warning('Skip 5s PPG comparison because the signal is empty.')
                return

            window_len = max(1, int(round(window_seconds * float(fps))))
            if pred_signal.size > window_len:
                start = max(0, (pred_signal.size - window_len) // 2)
                end = start + window_len
                pred_signal = pred_signal[start:end]
                true_signal = true_signal[start:end]
                time_offset = start / float(fps)
            else:
                time_offset = 0.0

            pred_z = _zscore(pred_signal)
            true_z = _zscore(true_signal)
            time_axis = time_offset + np.arange(pred_z.size) / float(fps)
            corr = _safe_corr(pred_z, true_z)
            nrmse = _norm_rmse(pred_z, true_z)

            plt.figure(figsize=(10, 3.5))
            plt.plot(time_axis, true_z, label='true PPG (z-score)', color='tab:red', linestyle='--', linewidth=1.0)
            plt.plot(time_axis, pred_z, label='predicted PPG (z-score)', color='tab:blue', linewidth=1.1)
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized amplitude')
            plt.title(f'{title} | 5s window | corr={corr:.3f} nRMSE={nrmse:.3f}')
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(file_path, dpi=220, bbox_inches='tight')
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

        def save_subject_ppg_comparison_5s(subject_name, subject_records):
            subject_records = [
                item for item in sorted(subject_records, key=lambda item: item['file'])
                if item.get('true_ppg') is not None
            ]
            if not subject_records:
                logger.warning('No true PPG found for subject %s. Skip 5s PPG comparison.', subject_name)
                return

            candidate = subject_records[0]
            pred_sig = np.asarray(candidate['signal'], dtype=np.float32)
            true_sig = np.asarray(candidate['true_ppg'], dtype=np.float32)
            pred_sig, true_sig = _align_true_ppg_to_pred(pred_sig, true_sig)
            fps_val = float(candidate.get('fps', args.fps))

            window_len = max(1, int(round(5.0 * float(fps_val))))
            if pred_sig.size > window_len:
                start = max(0, (pred_sig.size - window_len) // 2)
                end = start + window_len
                pred_sig = pred_sig[start:end]
                true_sig = true_sig[start:end]
                time_offset = start / float(fps_val)
            else:
                time_offset = 0.0

            pred_z = _zscore(pred_sig)
            true_z = _zscore(true_sig)
            t = time_offset + np.arange(pred_z.size, dtype=np.float32) / fps_val
            corr = _safe_corr(pred_z, true_z)
            nrmse = _norm_rmse(pred_z, true_z)

            subject_dir = out_dir / subject_name
            subject_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t, true_z, color='tab:red', linestyle='--', linewidth=1.0, label='true PPG (z-score)')
            ax.plot(t, pred_z, color='tab:blue', linewidth=1.1, label='predicted PPG (z-score)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized amplitude')
            ax.set_title(f'Subject {subject_name} | 5s PPG comparison | corr={corr:.3f} nRMSE={nrmse:.3f}')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(subject_dir / f'{subject_name}_ppg_comparison_5s.png', dpi=220, bbox_inches='tight')
            plt.close(fig)

        def save_final_ppg_comparison(records_for_plot):
            """Save one report-ready predicted-vs-true PPG waveform comparison figure."""
            candidate = None
            for rec in records_for_plot:
                if rec.get('true_ppg') is not None:
                    candidate = rec
                    break
            if candidate is None:
                logger.warning('No true PPG found in records. Skip final PPG comparison figure.')
                return

            pred_sig = np.asarray(candidate['signal'], dtype=np.float32)
            true_sig = np.asarray(candidate['true_ppg'], dtype=np.float32)
            pred_sig, true_sig = _align_true_ppg_to_pred(pred_sig, true_sig)
            fps_val = float(candidate.get('fps', args.fps))
            t = np.arange(pred_sig.size, dtype=np.float32) / fps_val

            pred_z = _zscore(pred_sig)
            true_z = _zscore(true_sig)
            corr = _safe_corr(pred_z, true_z)
            nrmse = _norm_rmse(pred_z, true_z)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t, true_z, color='tab:red', linestyle='--', linewidth=1.0, label='true PPG (z-score)')
            ax.plot(t, pred_z, color='tab:blue', linewidth=1.1, label='predicted PPG (z-score)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized amplitude')
            ax.set_title(
                f"Final PPG comparison | {candidate['file']} | corr={corr:.3f} nRMSE={nrmse:.3f}"
            )
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(out_dir / 'ppg_final_comparison.png', dpi=220, bbox_inches='tight')
            plt.close(fig)

            pred_5s, true_5s, time_offset = pred_sig, true_sig, 0.0
            window_len = max(1, int(round(5.0 * float(fps_val))))
            if pred_sig.size > window_len:
                start = max(0, (pred_sig.size - window_len) // 2)
                end = start + window_len
                pred_5s = pred_sig[start:end]
                true_5s = true_sig[start:end]
                time_offset = start / float(fps_val)
            pred_5s_z = _zscore(pred_5s)
            true_5s_z = _zscore(true_5s)
            t5 = time_offset + np.arange(pred_5s_z.size, dtype=np.float32) / fps_val
            corr5 = _safe_corr(pred_5s_z, true_5s_z)
            nrmse5 = _norm_rmse(pred_5s_z, true_5s_z)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t5, true_5s_z, color='tab:red', linestyle='--', linewidth=1.0, label='true PPG (z-score)')
            ax.plot(t5, pred_5s_z, color='tab:blue', linewidth=1.1, label='predicted PPG (z-score)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized amplitude')
            ax.set_title(
                f"Final PPG comparison (5s window) | {candidate['file']} | corr={corr5:.3f} nRMSE={nrmse5:.3f}"
            )
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(out_dir / 'ppg_final_comparison_5s.png', dpi=220, bbox_inches='tight')
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
            data = torch.load(ds.files[k], weights_only=False)
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

            true_ppg = data.get('ppg')
            if true_ppg is not None:
                save_ppg_comparison_plot(
                    pred_signal=sig,
                    true_signal=np.asarray(true_ppg),
                    fps=fps_val,
                    file_path=subject_dir / f'ppg_comparison_example_{k:02d}.png',
                    title=f'PPG comparison | {title}',
                )
                save_ppg_comparison_plot_5s(
                    pred_signal=sig,
                    true_signal=np.asarray(true_ppg),
                    fps=fps_val,
                    file_path=subject_dir / f'ppg_comparison_5s_example_{k:02d}.png',
                    title=f'PPG comparison | {title}',
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
            save_subject_ppg_comparison_5s(subject_name, subject_groups[subject_name])

        save_final_ppg_comparison(records)

        logger.info('Saved evaluation outputs to %s', out_dir)
    except Exception:
        import traceback
        logger.warning('Plot generation failed:')
        traceback.print_exc()
        logger.warning('matplotlib not available — skipping plots')


if __name__ == '__main__':
    main()
