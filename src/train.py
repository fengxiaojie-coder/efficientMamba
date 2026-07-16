"""Training script for EfficientMamba PPG waveform prediction.

Usage (quick run):
    python -m src.train --clips_dir data/ubfc_clips --epochs 2 --batch_size 4 --max_samples 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import time
from collections import Counter
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import torch
from torch import nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

from src.datasets.ubfc_dataset import UBFCClipDataset
from src.models.efficientphys_mamba import EfficientPhysMambaRegressor, mamba_backend_status

# optional fast progress bars; fallback to no-op if tqdm not installed
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
import csv


def _sample_clip_temporal_lengths(clip_roots: list[Path], max_files: int = 256) -> dict[int, int]:
    """Return a sampled histogram of clip temporal lengths across clip roots."""
    counts: Counter[int] = Counter()
    inspected = 0
    for root in clip_roots:
        for fn in sorted(root.glob('**/*.pt')):
            if inspected >= max_files:
                return dict(sorted(counts.items()))
            try:
                d = torch.load(fn, weights_only=False)
                t = int(d['clip'].shape[0])
                counts[t] += 1
                inspected += 1
            except Exception:
                continue
    return dict(sorted(counts.items()))


def _neg_pearson_loss(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return 1-r where r is Pearson correlation over the current batch."""
    x = preds.reshape(-1)
    y = target.reshape(-1)
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y) + eps)
    if torch.isfinite(denom) and float(denom.item()) > eps:
        r = torch.sum(x * y) / denom
        return 1.0 - r
    # If variance is near zero, predictions are effectively constant.
    # Returning 0 here creates a dead-zone where Pearson provides no gradient signal.
    # Use a fixed penalty so the model is pushed away from collapsed outputs.
    return preds.new_tensor(1.0)


def _pearson_corr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Return Pearson correlation coefficient r for the flattened tensors."""
    x = preds.reshape(-1)
    y = target.reshape(-1)
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y) + eps)
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return float('nan')
    return float((torch.sum(x * y) / denom).item())


def _align_by_lag_torch(preds: torch.Tensor, target: torch.Tensor, lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Align [B,T,1] waveform tensors by integer frame lag.

    lag > 0 means preds are shifted forward relative to target.
    """
    if lag == 0:
        return preds, target
    t = preds.size(1)
    if lag > 0:
        if lag >= t:
            return preds[:, :0, :], target[:, :0, :]
        return preds[:, lag:, :], target[:, :-lag, :]
    shift = -lag
    if shift >= t:
        return preds[:, :0, :], target[:, :0, :]
    return preds[:, :-shift, :], target[:, shift:, :]


def _align_by_lag_numpy(preds: np.ndarray, target: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Align [N,T] waveform arrays by integer frame lag."""
    if lag == 0:
        return preds, target
    t = preds.shape[1]
    if lag > 0:
        if lag >= t:
            return preds[:, :0], target[:, :0]
        return preds[:, lag:], target[:, :-lag]
    shift = -lag
    if shift >= t:
        return preds[:, :0], target[:, :0]
    return preds[:, :-shift], target[:, shift:]


def _safe_pearson_numpy(preds: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson r over flattened arrays with finite checks."""
    x = preds.reshape(-1)
    y = target.reshape(-1)
    if x.size <= 1 or y.size <= 1:
        return float('nan')
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else float('nan')


def _best_lag_pearson_numpy(preds: np.ndarray, target: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Find lag in [-max_lag, max_lag] that maximizes Pearson r."""
    max_lag = int(max(0, max_lag))
    best_lag = 0
    best_corr = _safe_pearson_numpy(preds, target)

    if max_lag <= 0:
        return best_lag, best_corr

    for lag in range(-max_lag, max_lag + 1):
        p_aligned, t_aligned = _align_by_lag_numpy(preds, target, lag)
        if p_aligned.shape[1] < 3:
            continue
        cur_corr = _safe_pearson_numpy(p_aligned, t_aligned)
        if not np.isfinite(cur_corr):
            continue
        if (not np.isfinite(best_corr)) or (cur_corr > best_corr) or (
            abs(cur_corr - best_corr) <= 1e-9 and abs(lag) < abs(best_lag)
        ):
            best_lag = lag
            best_corr = cur_corr

    return best_lag, best_corr


def _best_lag_pearson_per_clip_numpy(
    preds: np.ndarray,
    target: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find per-clip best lag and Pearson r for [N,T] arrays."""
    max_lag = int(max(0, max_lag))
    n = int(preds.shape[0])
    best_lags = np.zeros((n,), dtype=np.int32)
    best_corrs = np.full((n,), np.nan, dtype=np.float64)

    for i in range(n):
        p = preds[i:i + 1]
        t = target[i:i + 1]
        best_lag = 0
        best_corr = _safe_pearson_numpy(p, t)

        if max_lag > 0:
            for lag in range(-max_lag, max_lag + 1):
                p_aligned, t_aligned = _align_by_lag_numpy(p, t, lag)
                if p_aligned.shape[1] < 3:
                    continue
                cur_corr = _safe_pearson_numpy(p_aligned, t_aligned)
                if not np.isfinite(cur_corr):
                    continue
                if (not np.isfinite(best_corr)) or (cur_corr > best_corr) or (
                    abs(cur_corr - best_corr) <= 1e-9 and abs(lag) < abs(best_lag)
                ):
                    best_lag = lag
                    best_corr = cur_corr

        best_lags[i] = int(best_lag)
        best_corrs[i] = float(best_corr) if np.isfinite(best_corr) else np.nan

    return best_lags, best_corrs


def _format_lag_distribution(lag_counts: dict[int, int]) -> str:
    """Format lag histogram as a compact string like -1:12|0:88|1:9."""
    if not lag_counts:
        return 'none'
    return '|'.join(f'{int(k)}:{int(v)}' for k, v in sorted(lag_counts.items(), key=lambda kv: kv[0]))


def _lag_mode_from_counts(lag_counts: dict[int, int]) -> int:
    """Return modal lag; ties prefer smaller absolute lag then smaller lag."""
    if not lag_counts:
        return 0
    max_count = max(lag_counts.values())
    candidates = [k for k, v in lag_counts.items() if v == max_count]
    return int(sorted(candidates, key=lambda x: (abs(x), x))[0])


def _best_lag_base_loss_per_clip(
    preds: torch.Tensor,
    target: torch.Tensor,
    base_loss_fn,
    max_lag: int,
) -> tuple[list[int], torch.Tensor]:
    """Compute per-clip best lag by minimum loss and return mean best loss."""
    max_lag = int(max(0, max_lag))
    bsz = int(preds.size(0))
    best_lags: list[int] = []
    best_losses: list[torch.Tensor] = []

    for bi in range(bsz):
        p_i = preds[bi:bi + 1]
        t_i = target[bi:bi + 1]

        best_lag = 0
        best_loss = base_loss_fn(p_i, t_i)
        best_score = float(best_loss.detach().item())

        if max_lag > 0:
            for lag in range(-max_lag, max_lag + 1):
                p_aligned, t_aligned = _align_by_lag_torch(p_i, t_i, lag)
                if p_aligned.size(1) < 3:
                    continue
                cur_loss = base_loss_fn(p_aligned, t_aligned)
                cur_score = float(cur_loss.detach().item())
                if (cur_score < best_score) or (abs(cur_score - best_score) <= 1e-9 and abs(lag) < abs(best_lag)):
                    best_score = cur_score
                    best_loss = cur_loss
                    best_lag = lag

        best_lags.append(int(best_lag))
        best_losses.append(best_loss)

    if not best_losses:
        return best_lags, preds.new_tensor(float('nan'))
    return best_lags, torch.stack(best_losses).mean()


def _soft_lag_base_loss_per_clip(
    preds: torch.Tensor,
    target: torch.Tensor,
    base_loss_fn,
    max_lag: int,
    temperature: float,
) -> tuple[list[float], torch.Tensor]:
    """Compute per-clip soft lag expectation and softmax-weighted mean loss."""
    max_lag = int(max(0, max_lag))
    temp = float(max(1e-6, temperature))
    bsz = int(preds.size(0))
    soft_lags: list[float] = []
    soft_losses: list[torch.Tensor] = []

    for bi in range(bsz):
        p_i = preds[bi:bi + 1]
        t_i = target[bi:bi + 1]

        lag_values: list[int] = []
        loss_terms: list[torch.Tensor] = []

        if max_lag <= 0:
            lag_values = [0]
            loss_terms = [base_loss_fn(p_i, t_i)]
        else:
            for lag in range(-max_lag, max_lag + 1):
                p_aligned, t_aligned = _align_by_lag_torch(p_i, t_i, lag)
                if p_aligned.size(1) < 3:
                    continue
                lag_values.append(int(lag))
                loss_terms.append(base_loss_fn(p_aligned, t_aligned))

            if not loss_terms:
                lag_values = [0]
                loss_terms = [base_loss_fn(p_i, t_i)]

        loss_stack = torch.stack(loss_terms)
        logits = -torch.stack([loss.detach() for loss in loss_terms]) / temp
        weights = torch.softmax(logits, dim=0)
        soft_loss = torch.sum(weights * loss_stack)
        lag_tensor = torch.tensor(lag_values, device=preds.device, dtype=preds.dtype)
        soft_lag = torch.sum(weights * lag_tensor)

        soft_lags.append(float(soft_lag.item()))
        soft_losses.append(soft_loss)

    if not soft_losses:
        return soft_lags, preds.new_tensor(float('nan'))
    return soft_lags, torch.stack(soft_losses).mean()


def _best_lag_base_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    base_loss_fn,
    max_lag: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find lag with minimum base loss and return aligned tensors and best loss."""
    max_lag = int(max(0, max_lag))
    best_lag = 0
    best_preds = preds
    best_target = target
    best_loss = base_loss_fn(preds, target)
    best_score = float(best_loss.detach().item())

    if max_lag <= 0:
        return best_lag, best_preds, best_target, best_loss

    for lag in range(-max_lag, max_lag + 1):
        p_aligned, t_aligned = _align_by_lag_torch(preds, target, lag)
        if p_aligned.size(1) < 3:
            continue
        cur_loss = base_loss_fn(p_aligned, t_aligned)
        cur_score = float(cur_loss.detach().item())
        if (cur_score < best_score) or (abs(cur_score - best_score) <= 1e-9 and abs(lag) < abs(best_lag)):
            best_score = cur_score
            best_loss = cur_loss
            best_lag = lag
            best_preds = p_aligned
            best_target = t_aligned

    return best_lag, best_preds, best_target, best_loss


def _build_roi_map(clips, roi_types, device):
    """Build data-driven soft ROI prior for non-full ROI samples.

    For each sample, estimate ROI support from non-black raw pixels,
    aggregate over time, then smooth into a soft map in [0, 1].
    """
    if roi_types is None:
        return None
    try:
        B = clips.size(0)
        H = clips.size(-2)
        W = clips.size(-1)
        mask = torch.ones((B, 1, H, W), device=device)

        # Prefer raw channels (3:6) for ROI support when available.
        # clips shape: [B, T, 6, H, W] in normal training.
        if clips.size(2) >= 6:
            raw = clips[:, :, 3:6, :, :]
        else:
            raw = clips[:, :, :3, :, :]

        # Binary ROI support per frame, then temporal mean.
        support_t = (raw.abs().sum(dim=2) > 1e-6).float()  # [B, T, H, W]
        support = support_t.mean(dim=1, keepdim=True)      # [B, 1, H, W]

        # Smooth into a soft prior to avoid hard-edge bias.
        support = F.avg_pool2d(support, kernel_size=5, stride=1, padding=2)
        support = F.avg_pool2d(support, kernel_size=5, stride=1, padding=2)
        support = torch.clamp(support, 0.0, 1.0)

        # Keep a small floor so the prior guides but doesn't collapse attention.
        soft_support = 0.15 + 0.85 * support

        for i, rt in enumerate(roi_types):
            if isinstance(rt, bytes):
                rt = rt.decode()
            if rt is None or rt == 'full':
                mask[i, 0] = 1.0
            else:
                mask[i, 0] = soft_support[i, 0]
        return mask
    except Exception:
        return None


def _suppress_edge_diff_outliers(
    clips: torch.Tensor,
    diff_edge_abs_max: float = 0.0,
    edge_width: int = 1,
) -> torch.Tensor:
    """Zero diff-channel outliers on ROI edge pixels.

    clips shape: [B, T, 6, H, W], where channels 0:3 are diff and 3:6 are raw.
    Edge is estimated from non-black raw ROI mask using morphological gradient.
    """
    thr = float(diff_edge_abs_max)
    if thr <= 0.0:
        return clips

    w = int(max(1, edge_width))
    B, T, C, H, W = clips.shape
    if C < 6:
        return clips

    raw = clips[:, :, 3:6, :, :]
    face_mask = (raw.abs().sum(dim=2, keepdim=True) > 1e-6).float()  # [B,T,1,H,W]

    face_bt = face_mask.reshape(B * T, 1, H, W)
    k = 2 * w + 1
    dil = F.max_pool2d(face_bt, kernel_size=k, stride=1, padding=w)
    ero = 1.0 - F.max_pool2d(1.0 - face_bt, kernel_size=k, stride=1, padding=w)
    edge_bt = (dil - ero) > 0.0
    edge = edge_bt.reshape(B, T, 1, H, W)

    diff = clips[:, :, 0:3, :, :]
    outlier = diff.abs() > thr
    suppress = outlier & edge.expand_as(outlier)
    if suppress.any():
        clips = clips.clone()
        clips[:, :, 0:3, :, :] = diff.masked_fill(suppress, 0.0)
    return clips


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())
    return total ** 0.5


def _save_step_debug_visual(
    save_dir: Path,
    epoch: int,
    step: int,
    clips: torch.Tensor,
    preds: torch.Tensor,
    hrs: torch.Tensor,
    loss_value: float,
    grad_norm: float,
    attn_dict: dict | None,
):
    """Save per-step debug panel and numeric diagnostics for quick failure checks."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn.functional as F

        debug_dir = save_dir / 'debug_steps'
        debug_dir.mkdir(parents=True, exist_ok=True)

        c = clips[0].detach().cpu()  # [T,6,H,W]
        t_mid = c.shape[0] // 2
        raw = c[t_mid, 3:6].permute(1, 2, 0).numpy()
        raw = np.clip(raw, 0.0, 1.0)
        diff = c[t_mid, :3].permute(1, 2, 0).numpy()
        diff_mag = np.linalg.norm(diff, axis=-1)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        axes[0].imshow(raw)
        axes[0].set_title('raw RGB (mid frame)')
        axes[0].axis('off')

        im1 = axes[1].imshow(diff_mag, cmap='magma')
        axes[1].set_title('diff magnitude')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        if attn_dict is not None:
            g1 = attn_dict['g1'][0, t_mid, 0].detach().cpu().numpy()
            g2_t = min(t_mid, attn_dict['g2'].shape[1] - 1)
            g2 = attn_dict['g2'][0, g2_t, 0].detach().cpu().unsqueeze(0).unsqueeze(0)
            g2_up = F.interpolate(g2, size=(g1.shape[0], g1.shape[1]), mode='bilinear', align_corners=False)
            g2_up = g2_up.squeeze().numpy()

            im2 = axes[2].imshow(g1, cmap='hot')
            axes[2].set_title('G1 attention')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            im3 = axes[3].imshow(g2_up, cmap='hot')
            axes[3].set_title('G2 attention (upsampled)')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
        else:
            axes[2].axis('off')
            axes[2].set_title('G1 attention (N/A)')
            axes[3].axis('off')
            axes[3].set_title('G2 attention (N/A)')

        pred_val = float(preds[0].item())
        true_val = float(hrs[0].item())
        fig.suptitle(
            f'E{epoch} S{step} | loss={loss_value:.4f} pred={pred_val:.2f} true={true_val:.2f} grad={grad_norm:.3e}',
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(debug_dir / f'epoch_{epoch:03d}_step_{step:05d}.png', dpi=120)
        plt.close(fig)

        csv_path = debug_dir / 'debug_step_metrics.csv'
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(['epoch', 'step', 'loss', 'pred', 'true', 'grad_norm'])
            w.writerow([epoch, step, loss_value, pred_val, true_val, grad_norm])
    except Exception as e:
        logger.warning('Could not save step debug visualization at epoch=%d step=%d: %s', epoch, step, e)


def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    neg_pearson_coef: float = 0.0,
    mae_coef: float = 1.0,  # reserved for backward compatibility
    desc: str | None = None,
    epoch: int = 0,
    save_dir: Path | None = None,
    debug_viz_every_steps: int = 0,
    diff_edge_abs_max: float = 0.0,
    diff_edge_width: int = 1,
    train_max_lag_frames: int = 0,
    train_shift_invariant_weight: float = 0.0,
    train_lag_temperature: float = 1.0,
):
    model.train()
    total_loss = 0.0
    count = 0
    pearson_sum = 0.0
    pearson_count = 0
    iterator = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(iterator, start=1):
        # support datasets that may return ROI metadata: (clips, ppg_waveform[, roi_type, roi_bbox])
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            clips = batch[0]
            ppg = batch[1]  # PPG waveform [B, T] - real ground truth
            roi_types = batch[2] if len(batch) > 2 else None
        else:
            clips, ppg = batch
            roi_types = None

        clips = clips.to(device)  # [B, T, 6, H, W]
        clips = _suppress_edge_diff_outliers(
            clips,
            diff_edge_abs_max=float(diff_edge_abs_max),
            edge_width=int(diff_edge_width),
        )
        ppg = ppg.to(device)  # [B, T]

        # Reshape PPG to match model output: [B, T] -> [B, T, 1]
        if ppg.dim() == 1:
            ppg = ppg.unsqueeze(0).unsqueeze(-1)  # [T] -> [1, T, 1]
        elif ppg.dim() == 2:
            ppg = ppg.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        # build a data-driven soft ROI map per-sample when explicit ROI was used
        roi_map = _build_roi_map(clips, roi_types, device)

        optimizer.zero_grad()
        preds = model(clips, roi_map=roi_map) if roi_map is not None else model(clips)  # [B, T, 1]

        # Optimize configured objective with optional shift-invariant alignment.
        si_w = float(np.clip(train_shift_invariant_weight, 0.0, 1.0))
        max_lag = int(max(0, train_max_lag_frames))

        base_loss_zero = loss_fn(preds, ppg)
        if si_w > 0.0 and max_lag > 0:
            _lags_base, base_loss_si = _soft_lag_base_loss_per_clip(
                preds,
                ppg,
                loss_fn,
                max_lag=max_lag,
                temperature=float(train_lag_temperature),
            )
            base_loss = (1.0 - si_w) * base_loss_zero + si_w * base_loss_si
        else:
            base_loss = base_loss_zero

        if float(neg_pearson_coef) > 0.0:
            pearson_penalty_zero = _neg_pearson_loss(preds, ppg)
            if si_w > 0.0 and max_lag > 0:
                _lags_pearson, pearson_penalty_si = _soft_lag_base_loss_per_clip(
                    preds,
                    ppg,
                    _neg_pearson_loss,
                    max_lag=max_lag,
                    temperature=float(train_lag_temperature),
                )
                pearson_penalty = (1.0 - si_w) * pearson_penalty_zero + si_w * pearson_penalty_si
            else:
                pearson_penalty = pearson_penalty_zero
            loss = base_loss + float(neg_pearson_coef) * pearson_penalty
        else:
            loss = base_loss
        batch_pearson = _pearson_corr(preds.detach(), ppg.detach())
        if np.isfinite(batch_pearson):
            pearson_sum += batch_pearson * clips.size(0)
            pearson_count += clips.size(0)
        loss.backward()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()

        if debug_viz_every_steps > 0 and save_dir is not None and (step % debug_viz_every_steps == 0):
            attn_dict = None
            try:
                model.eval()
                with torch.no_grad():
                    clip_dbg = clips[:1]
                    roi_dbg = roi_map[:1] if roi_map is not None else None
                    _, attn_dict = model(clip_dbg, roi_map=roi_dbg, return_attention=True)
            except Exception as e:
                logger.warning('Failed to fetch attention for debug viz at epoch=%d step=%d: %s', epoch, step, e)
            finally:
                model.train()

            _save_step_debug_visual(
                save_dir=save_dir,
                epoch=epoch,
                step=step,
                clips=clips,
                preds=preds,
                hrs=ppg,  # Now ppg waveform, not scalar HR
                loss_value=float(loss.item()),
                grad_norm=float(grad_norm),
                attn_dict=attn_dict,
            )

        total_loss += loss.item() * clips.size(0)
        count += clips.size(0)
    return total_loss / max(1, count), (pearson_sum / max(1, pearson_count) if pearson_count > 0 else float('nan'))


def eval_model(
    model,
    loader,
    device,
    desc: str | None = None,
    max_batches: int = 0,
    loss_fn: nn.Module | None = None,
    neg_pearson_coef: float = 0.0,
    diff_edge_abs_max: float = 0.0,
    diff_edge_width: int = 1,
    train_max_lag_frames: int = 0,
    train_shift_invariant_weight: float = 0.0,
    train_lag_temperature: float = 1.0,
):
    model.eval()
    import numpy as np
    preds_all = []
    ppgs_all = []
    eval_loss_sum = 0.0
    eval_count = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc=desc, leave=False)
        for batch_idx, batch in enumerate(iterator):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                clips = batch[0]
                ppg = batch[1]  # PPG waveform [B, T]
                roi_types = batch[2] if len(batch) > 2 else None
            else:
                clips, ppg = batch
                roi_types = None

            clips = clips.to(device)
            clips = _suppress_edge_diff_outliers(
                clips,
                diff_edge_abs_max=float(diff_edge_abs_max),
                edge_width=int(diff_edge_width),
            )
            if ppg.dim() == 2:
                ppg = ppg.unsqueeze(-1)  # [B, T] -> [B, T, 1]
            ppg = ppg.to(device)

            roi_map = _build_roi_map(clips, roi_types, device)

            out = model(clips, roi_map=roi_map) if roi_map is not None else model(clips)  # [B, T, 1]
            preds_all.append(out.cpu().numpy())
            ppgs_all.append(ppg.cpu().numpy())
            if loss_fn is not None:
                si_w = float(np.clip(train_shift_invariant_weight, 0.0, 1.0))
                max_lag = int(max(0, train_max_lag_frames))

                base_loss_zero = loss_fn(out, ppg)
                if si_w > 0.0 and max_lag > 0:
                    _lags_base, base_loss_si = _soft_lag_base_loss_per_clip(
                        out,
                        ppg,
                        loss_fn,
                        max_lag=max_lag,
                        temperature=float(train_lag_temperature),
                    )
                    base_loss = (1.0 - si_w) * base_loss_zero + si_w * base_loss_si
                else:
                    base_loss = base_loss_zero

                if float(neg_pearson_coef) > 0.0:
                    pearson_penalty_zero = _neg_pearson_loss(out, ppg)
                    if si_w > 0.0 and max_lag > 0:
                        _lags_pearson, pearson_penalty_si = _soft_lag_base_loss_per_clip(
                            out,
                            ppg,
                            _neg_pearson_loss,
                            max_lag=max_lag,
                            temperature=float(train_lag_temperature),
                        )
                        pearson_penalty = (1.0 - si_w) * pearson_penalty_zero + si_w * pearson_penalty_si
                    else:
                        pearson_penalty = pearson_penalty_zero
                    batch_loss = base_loss + float(neg_pearson_coef) * pearson_penalty
                else:
                    batch_loss = base_loss
                eval_loss_sum += float(batch_loss.item()) * clips.size(0)
                eval_count += clips.size(0)
    if not preds_all:
        return float('nan'), {
            'pred_mean': float('nan'),
            'pred_std': float('nan'),
            'pred_min': float('nan'),
            'pred_max': float('nan'),
            'true_mean': float('nan'),
            'true_std': float('nan'),
        }, float('nan'), float('nan'), {
            'hard_pearson': float('nan'),
            'hard_mode': 0,
            'hard_mean': 0.0,
            'hard_median': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'sample_100': [],
        }
    preds = np.vstack(preds_all)
    ppgs = np.vstack(ppgs_all)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds[..., 0]
    if ppgs.ndim == 3 and ppgs.shape[-1] == 1:
        ppgs = ppgs[..., 0]
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]
    if ppgs.ndim == 1:
        ppgs = ppgs[np.newaxis, :]
    # Compute frame-wise MSE
    mse = float(np.mean((preds - ppgs) ** 2))
    pred_stats = {
        'pred_mean': float(np.mean(preds)),
        'pred_std': float(np.std(preds)),
        'pred_min': float(np.min(preds)),
        'pred_max': float(np.max(preds)),
        'true_mean': float(np.mean(ppgs)),
        'true_std': float(np.std(ppgs)),
    }
    pearson = _safe_pearson_numpy(preds, ppgs)
    lag_search = int(max(0, train_max_lag_frames))
    if lag_search > 0:
        hard_lags, hard_corrs = _best_lag_pearson_per_clip_numpy(preds, ppgs, max_lag=lag_search)
        hard_counts = dict(sorted(Counter(int(v) for v in hard_lags.tolist()).items()))
        hard_finite = np.isfinite(hard_corrs)
        hard_pearson = float(np.mean(hard_corrs[hard_finite])) if np.any(hard_finite) else float('nan')
        hard_mode = _lag_mode_from_counts(hard_counts)
        hard_mean = float(np.mean(hard_lags)) if hard_lags.size > 0 else 0.0
        hard_median = float(np.median(hard_lags)) if hard_lags.size > 0 else 0.0

        preds_t = torch.tensor(preds[..., np.newaxis], dtype=torch.float32)
        ppgs_t = torch.tensor(ppgs[..., np.newaxis], dtype=torch.float32)
        soft_lags, soft_pearson_loss = _soft_lag_base_loss_per_clip(
            preds_t,
            ppgs_t,
            _neg_pearson_loss,
            max_lag=lag_search,
            temperature=float(train_lag_temperature),
        )
        pearson_lag = float(1.0 - soft_pearson_loss.detach().item()) if torch.isfinite(soft_pearson_loss) else float('nan')
        lag_mean = float(np.mean(soft_lags)) if soft_lags else 0.0
        lag_median = float(np.median(soft_lags)) if soft_lags else 0.0
        lag_sample_100 = [float(v) for v in soft_lags[: min(100, len(soft_lags))]]
    else:
        pearson_lag = pearson
        hard_pearson = pearson
        hard_mode = 0
        hard_mean = 0.0
        hard_median = 0.0
        lag_mean = 0.0
        lag_median = 0.0
        lag_sample_100 = [0.0 for _ in range(min(100, int(preds.shape[0])))]
    lag_stats = {
        'hard_pearson': float(hard_pearson),
        'hard_mode': int(hard_mode),
        'hard_mean': float(hard_mean),
        'hard_median': float(hard_median),
        'mean': float(lag_mean),
        'median': float(lag_median),
        'sample_100': lag_sample_100,
    }
    eval_loss = float(eval_loss_sum / max(1, eval_count)) if loss_fn is not None else float('nan')
    return mse, pred_stats, eval_loss, pearson, pearson_lag, lag_stats


def main():
    default_weight_decay = 1e-4
    default_use_augment = True
    default_warmup_epochs = 5
    default_cosine_epochs = 10
    default_min_lr = 1e-5

    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_dir', default='data/ubfc_clips')
    parser.add_argument('--extra_clips_dirs', type=str, default='',
                        help='Comma-separated extra clip directories to merge into training (e.g. data/ubfc_phys_clips,data/other_clips)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frame_depth', type=int, default=128,
                        help='Temporal clip length expected by TSM front-end (should match preprocessing clip_len)')
    parser.add_argument('--skip_frame_depth_check', action='store_true',
                        help='Skip per-file frame-depth validation at dataset load for faster startup. Use only when all clips share the same T.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=default_weight_decay,
                        help='Weight decay for AdamW. Lower values can help avoid mean-collapse.')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable training-time augmentation for debugging collapse issues')
    parser.add_argument('--loss', type=str, default='negative_pearson', choices=['mse', 'huber', 'negative_pearson'],
                        help='Base loss type: mse/huber or negative_pearson (EfficientPhys-style)')
    parser.add_argument('--huber_delta', type=float, default=3.0,
                        help='Delta used when --loss huber')
    parser.add_argument('--neg_pearson_coef', type=float, default=0.1,
                        help='Add negative Pearson term: total_loss = base_loss + coef * (1-r), where r is batch Pearson corr')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Stop training if val_pearson does not improve for N epochs. Set to 0 to disable.')
    parser.add_argument('--two_stage_training', action='store_true',
                        help='Enable two-stage training: stage1 (light MAE, heavy Pearson) then stage2 (full MAE+Pearson)')
    parser.add_argument('--stage1_epochs', type=int, default=10,
                        help='Number of epochs for stage1 (with light MAE). Total epochs must be > stage1_epochs')
    parser.add_argument('--stage1_mae_coef', type=float, default=0.01,
                        help='MAE coefficient in stage1 (e.g., 0.01 or 0.0 for Pearson-only)')
    parser.add_argument('--stage2_mae_coef', type=float, default=1.0,
                        help='MAE coefficient in stage2 (e.g., 1.0 for full MAE)')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='If >0, limit dataset to this many samples for quick runs')
    parser.add_argument('--max_subjects', type=int, default=0,
                        help='If >0, limit dataset to this many subjects (subject prefix before "_clip_")')
    parser.add_argument('--subjects', type=str, default='',
                        help='Comma-separated list of subject names to include (e.g. subject01,subject02)')
    parser.add_argument('--per_subject_limit', type=int, default=0,
                        help='If >0, limit number of clips per subject to this many')
    parser.add_argument('--split_mode', type=str, default='clip_random',
                        choices=['clip_random', 'subject_disjoint'],
                        help='How to split train/val: random by clip, or disjoint by subject')
    parser.add_argument('--temporal_backbone', type=str, default='auto', choices=['auto', 'mamba', 'gru'],
                        help='Temporal model selection: auto=prefer mamba with GRU fallback, mamba=fail if unavailable, gru=force GRU')
    parser.add_argument('--debug_viz_every_steps', type=int, default=0,
                        help='If >0, save quick debug visualizations every N training steps to checkpoints/debug_steps')
    parser.add_argument('--train_eval_batches', type=int, default=10,
                        help='Number of batches to sample for train-set eval each epoch (0=full, default=10 for speed)')
    parser.add_argument('--diff_edge_abs_max', type=float, default=0.0,
                        help='If >0, suppress diff-channel pixels with |diff| above threshold on ROI edge regions only.')
    parser.add_argument('--diff_edge_width', type=int, default=1,
                        help='ROI edge width in pixels used with --diff_edge_abs_max.')
    parser.add_argument('--train_max_lag_frames', type=int, default=0,
                        help='If >0, enable shift-invariant training loss by searching lag in [-N, N] frames.')
    parser.add_argument('--train_shift_invariant_weight', type=float, default=0.0,
                        help='Blend weight for shift-invariant objective in [0,1]. 0=disabled, 1=fully shift-invariant.')
    parser.add_argument('--train_lag_temperature', type=float, default=1.0,
                        help='Softmax temperature for lag weighting. Lower values make the lag distribution sharper.')
    args = parser.parse_args()

    # configure logging early so modules emit consistent messages
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_augment = default_use_augment and (not args.no_augment)
    logger.info('Using device: %s', device)
    logger.info('Training config: augment=%s, weight_decay=%.1e, loss=%s, neg_pearson_coef=%.3f',
                use_augment, args.weight_decay, args.loss, args.neg_pearson_coef)
    if float(args.diff_edge_abs_max) > 0.0:
        logger.info('Edge diff suppression enabled: abs_max=%.4f, edge_width=%d',
                    float(args.diff_edge_abs_max), int(args.diff_edge_width))
    if int(args.train_max_lag_frames) > 0 and float(args.train_shift_invariant_weight) > 0.0:
        logger.info(
            'Shift-invariant loss enabled: max_lag=%d frames, weight=%.3f, temperature=%.3f',
            int(args.train_max_lag_frames),
            float(args.train_shift_invariant_weight),
            float(args.train_lag_temperature),
        )
    if args.two_stage_training:
        logger.info('Two-stage training ENABLED: stage1=%d epochs (mae_coef=%.3f), stage2 (mae_coef=%.3f)',
                    args.stage1_epochs, args.stage1_mae_coef, args.stage2_mae_coef)
    logger.info(
        'LR schedule: base_lr=%.1e, warmup_epochs=%d, cosine_epochs=%d, min_lr=%.1e',
        args.lr,
        default_warmup_epochs,
        default_cosine_epochs,
        default_min_lr,
    )

    clip_roots = [Path(args.clips_dir)]
    if args.extra_clips_dirs:
        extra_dirs = [Path(p.strip()) for p in args.extra_clips_dirs.split(',') if p.strip()]
        clip_roots.extend(extra_dirs)
    for root in clip_roots:
        if not root.exists():
            raise SystemExit(f'Clip directory does not exist: {root}')

    dataset_frame_depth = None if args.skip_frame_depth_check else args.frame_depth
    if args.skip_frame_depth_check:
        logger.warning(
            'Skipping frame_depth validation during dataset indexing for speed. '
            'Ensure all clips have T=%d, or DataLoader/model shape errors may occur.',
            args.frame_depth,
        )

    ds = UBFCClipDataset(clip_roots[0], frame_depth=dataset_frame_depth)
    merged_files = list(ds.files)
    for root in clip_roots[1:]:
        extra_ds = UBFCClipDataset(root, frame_depth=dataset_frame_depth)
        merged_files.extend(extra_ds.files)
    ds.files = sorted(merged_files)
    logger.info('Loaded %d clips from %d directory(ies): %s', len(ds.files), len(clip_roots), [str(p) for p in clip_roots])
    if len(ds.files) == 0:
        # Provide actionable mismatch hints when all clips are filtered by frame_depth.
        t_hist = _sample_clip_temporal_lengths(clip_roots, max_files=256)
        if t_hist:
            t_values = list(t_hist.keys())
            suggested = max(t_hist.items(), key=lambda kv: kv[1])[0]
            raise SystemExit(
                'No clips found after frame_depth filtering. '
                f'Current --frame_depth={args.frame_depth}, sampled clip T distribution={t_hist}. '
                f'Try --frame_depth {suggested} (or re-run preprocessing with --clip_len {args.frame_depth}).'
            )

    # Subject-aware selection
    def subject_from_name(fn: Path) -> str:
        name = fn.name
        base_subject = name.split('_clip_', 1)[0] if '_clip_' in name else name
        try:
            fn_resolved = fn.resolve()
            for root in clip_roots:
                root_resolved = root.resolve()
                try:
                    fn_resolved.relative_to(root_resolved)
                    return f'{root_resolved.name}::{base_subject}'
                except ValueError:
                    continue
        except Exception:
            pass
        return base_subject

    # Build subject -> files mapping
    subj_map: dict = {}
    for fn in ds.files:
        s = subject_from_name(fn)
        subj_map.setdefault(s, []).append(fn)

    # Apply explicit subjects filter
    if args.subjects:
        want = [s.strip() for s in args.subjects.split(',') if s.strip()]
        subj_map = {s: subj_map[s] for s in subj_map if s in want}

    # Limit number of subjects
    subject_list = sorted(subj_map.keys())
    if args.max_subjects and args.max_subjects > 0:
        subject_list = subject_list[: args.max_subjects]

    # Apply per-subject clip limit
    filtered_files = []
    for s in subject_list:
        files = sorted(subj_map[s])
        if args.per_subject_limit and args.per_subject_limit > 0:
            files = files[: args.per_subject_limit]
        filtered_files.extend(files)

    # If user passed --max_samples and split_mode is clip_random, enforce it
    if args.max_samples > 0 and args.split_mode == 'clip_random':
        filtered_files = filtered_files[: args.max_samples]

    # Replace dataset files with filtered list
    ds.files = filtered_files
    n = len(ds)
    if n == 0:
        raise SystemExit('No clips found after subject/sample filtering. Check --subjects/--max_subjects/--per_subject_limit settings.')
    # split (manual so we can enable augment only for training set)
    if args.split_mode == 'clip_random':
        # shuffle and split by clip
        random.shuffle(filtered_files)
        val_n = max(int(0.2 * n), 1)
        train_files = filtered_files[:-val_n]
        val_files = filtered_files[-val_n:]
    else:
        # subject_disjoint: split by subject so train/val subjects don't overlap
        subj_map_filtered = {}
        for fn in ds.files:
            s = subject_from_name(fn)
            subj_map_filtered.setdefault(s, []).append(fn)
        subjects = sorted(subj_map_filtered.keys())
        random.seed(0)
        random.shuffle(subjects)
        # choose number of validation subjects; ensure at least one train subject
        if len(subjects) <= 1:
            val_subj_n = 0
        else:
            val_subj_n = max(int(0.2 * len(subjects)), 1)
        val_subjects = set(subjects[:val_subj_n])
        train_subjects = set(subjects[val_subj_n:])
        logger.info('Train subjects: %s', sorted(train_subjects))
        logger.info('Val subjects: %s', sorted(val_subjects))
        train_files = []
        val_files = []
        for s in subjects:
            files = sorted(subj_map_filtered[s])
            if s in val_subjects:
                val_files.extend(files)
            else:
                train_files.extend(files)
        # ensure we have at least one training file; if not, move one from val to train
        if len(train_files) == 0 and len(val_files) > 0:
            train_files.append(val_files.pop(0))

        train_ds = UBFCClipDataset(args.clips_dir)
        train_ds.files = train_files
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # create dataset instances: enable augment only for training dataset
    train_ds = UBFCClipDataset(args.clips_dir, augment=use_augment, frame_depth=dataset_frame_depth)
    train_ds.files = train_files
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )

    if len(val_files) > 0:
        val_ds = UBFCClipDataset(args.clips_dir, augment=False, frame_depth=dataset_frame_depth)
        val_ds.files = val_files
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
            pin_memory=True, persistent_workers=True,
        )
    else:
        val_loader = None

    use_mamba_requested = args.temporal_backbone != 'gru'
    model = EfficientPhysMambaRegressor(in_channels=6, frame_depth=args.frame_depth, use_mamba=use_mamba_requested)
    model.to(device)

    mamba_available, mamba_reason = mamba_backend_status()
    if args.temporal_backbone == 'mamba' and not model.temporal.uses_mamba:
        raise SystemExit(
            'Requested --temporal_backbone mamba, but Mamba backend is unavailable. '
            f'Reason: {mamba_reason}'
        )

    # Log model configuration
    backbone_name = 'Mamba' if model.temporal.uses_mamba else 'GRU'
    total_params = sum(p.numel() for p in model.parameters())
    temporal_params = sum(p.numel() for p in model.temporal.parameters())
    logger.info('=' * 70)
    logger.info('Model Configuration')
    logger.info('=' * 70)
    logger.info('Architecture: EfficientPhysMambaRegressor')
    logger.info('Temporal Backbone: %s (requested=%s, mamba_available=%s)', backbone_name, args.temporal_backbone, mamba_available)
    logger.info('Frame depth: %d, Input channels: 6', args.frame_depth)
    logger.info('Total parameters: %s', f'{total_params:,}')
    logger.info('Temporal backbone parameters: %s (%.1f%%)', f'{temporal_params:,}', 100*temporal_params/total_params)
    logger.info('Device: %s', device)
    logger.info('=' * 70)

    optimizer = torch_optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch_optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, default_cosine_epochs),
        eta_min=default_min_lr,
    )
    cosine_steps_done = 0
    if args.loss == 'negative_pearson':
        loss_fn = _neg_pearson_loss
    elif args.loss == 'huber':
        loss_fn = nn.HuberLoss(delta=float(args.huber_delta))
    else:
        loss_fn = nn.MSELoss()

    effective_neg_pearson_coef = float(args.neg_pearson_coef)
    if args.loss == 'negative_pearson' and effective_neg_pearson_coef > 0.0:
        logger.warning(
            'Ignoring --neg_pearson_coef=%.3f because --loss=negative_pearson already optimizes (1-r) directly.',
            effective_neg_pearson_coef,
        )
        effective_neg_pearson_coef = 0.0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'runs')) if SummaryWriter is not None else None

    best_val = float('-inf')
    epochs_done = []
    train_losses = []
    train_maes = []
    val_maes = []
    train_pearsons = []
    val_pearsons = []
    train_pred_stds = []
    val_pred_stds = []
    metrics_csv = save_dir / 'training_metrics.csv'
    lag_samples_csv = save_dir / 'lag_samples_every10_epochs.csv'
    # write header
    with open(metrics_csv, 'w', newline='') as _f:
        w = csv.writer(_f)
        w.writerow([
            'epoch',
            'train_loss',
            'train_mse',
            'val_mse',
            'val_loss',
            'train_pearson',
            'val_pearson',
            'train_pearson_lag',
            'val_pearson_lag',
            'train_pearson_lag_hard',
            'val_pearson_lag_hard',
            'train_hard_lag_mode',
            'val_hard_lag_mode',
            'train_soft_lag_mean',
            'val_soft_lag_mean',
            'train_soft_lag_median',
            'val_soft_lag_median',
            'train_soft_lag_100',
            'val_soft_lag_100',
            'train_pred_std',
            'val_pred_std',
        ])
    with open(lag_samples_csv, 'w', newline='') as _f:
        w = csv.writer(_f)
        w.writerow(['epoch', 'train_soft_lag_100', 'val_soft_lag_100'])

    # Early stopping counter
    patience_counter = 0
    early_stopping_patience = int(args.early_stopping_patience) if hasattr(args, 'early_stopping_patience') else 15

    for epoch in tqdm(range(1, args.epochs + 1), desc='Training', total=args.epochs, leave=True, ncols=100):
        current_lr = float(optimizer.param_groups[0]['lr'])
        t0 = time.time()
        if args.two_stage_training:
            if epoch <= args.stage1_epochs:
                mae_coef = float(args.stage1_mae_coef)
                stage_label = 'STAGE1(Pearson-heavy)'
            else:
                mae_coef = float(args.stage2_mae_coef)
                stage_label = 'STAGE2(MAE+Pearson)'
        else:
            mae_coef = 1.0
            stage_label = ''
        train_loss, train_pearson = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            neg_pearson_coef=effective_neg_pearson_coef,
            mae_coef=mae_coef,
            epoch=epoch,
            save_dir=save_dir,
            debug_viz_every_steps=int(args.debug_viz_every_steps),
            diff_edge_abs_max=float(args.diff_edge_abs_max),
            diff_edge_width=int(args.diff_edge_width),
            train_max_lag_frames=int(args.train_max_lag_frames),
            train_shift_invariant_weight=float(args.train_shift_invariant_weight),
            train_lag_temperature=float(args.train_lag_temperature),
            desc=f'Epoch {epoch}/{args.epochs} {stage_label} [train]',
        )
        # compute train MSE for monitoring (limited to train_eval_batches for speed)
        train_mse, train_stats, _train_eval_loss, train_eval_pearson, train_eval_pearson_lag, train_lag_stats = eval_model(
            model,
            train_loader,
            device,
            max_batches=int(args.train_eval_batches),
            desc=f'Epoch {epoch}/{args.epochs} [train-eval]',
            loss_fn=loss_fn,
            neg_pearson_coef=effective_neg_pearson_coef,
            diff_edge_abs_max=float(args.diff_edge_abs_max),
            diff_edge_width=int(args.diff_edge_width),
            train_max_lag_frames=int(args.train_max_lag_frames),
            train_shift_invariant_weight=float(args.train_shift_invariant_weight),
            train_lag_temperature=float(args.train_lag_temperature),
        )
        # compute val metrics if validation set exists
        if val_loader is not None:
            val_mse, val_stats, val_loss, val_pearson, val_pearson_lag, val_lag_stats = eval_model(
                model,
                val_loader,
                device,
                desc=f'Epoch {epoch}/{args.epochs} [val-eval]',
                loss_fn=loss_fn,
                neg_pearson_coef=effective_neg_pearson_coef,
                diff_edge_abs_max=float(args.diff_edge_abs_max),
                diff_edge_width=int(args.diff_edge_width),
                train_max_lag_frames=int(args.train_max_lag_frames),
                train_shift_invariant_weight=float(args.train_shift_invariant_weight),
                train_lag_temperature=float(args.train_lag_temperature),
            )
        else:
            val_mse = float('nan')
            val_loss = float('nan')
            val_pearson = float('nan')
            val_pearson_lag = float('nan')
            val_lag_stats = {
                'hard_pearson': float('nan'),
                'hard_mode': 0,
                'hard_mean': 0.0,
                'hard_median': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'sample_100': [],
            }
            val_stats = {
                'pred_std': float('nan'),
                'pred_mean': float('nan'),
                'pred_min': float('nan'),
                'pred_max': float('nan'),
                'true_std': float('nan'),
                'true_mean': float('nan'),
            }
        train_soft_lag_mean = float(train_lag_stats['mean'])
        val_soft_lag_mean = float(val_lag_stats['mean'])
        train_soft_lag_median = float(train_lag_stats['median'])
        val_soft_lag_median = float(val_lag_stats['median'])
        train_hard_lag_mode = int(train_lag_stats['hard_mode'])
        val_hard_lag_mode = int(val_lag_stats['hard_mode'])
        train_hard_pearson = float(train_lag_stats['hard_pearson'])
        val_hard_pearson = float(val_lag_stats['hard_pearson'])
        t1 = time.time()
        if args.two_stage_training:
            logger.info(
                'Epoch %d/%d [%s] - mae_coef=%.3f lr=%.2e train_loss=%.4f train_mse=%.4f val_mse=%.4f '
                'train_pearson=%.4f val_pearson=%.4f train_pearson_lag=%.4f val_pearson_lag=%.4f '
            'train_pearson_lag_hard=%.4f val_pearson_lag_hard=%.4f train_hard_lag_mode=%d val_hard_lag_mode=%d '
                'train_soft_lag_mean=%.4f val_soft_lag_mean=%.4f train_soft_lag_median=%.4f val_soft_lag_median=%.4f '
                'train_pred_std=%.4f val_pred_std=%.4f time=%.1fs',
                epoch,
                args.epochs,
                stage_label,
                mae_coef,
                current_lr,
                train_loss,
                train_mse,
                val_mse,
                train_eval_pearson,
                val_pearson,
                train_eval_pearson_lag,
                val_pearson_lag,
                train_hard_pearson,
                val_hard_pearson,
                train_hard_lag_mode,
                val_hard_lag_mode,
                train_soft_lag_mean,
                val_soft_lag_mean,
                train_soft_lag_median,
                val_soft_lag_median,
                train_stats['pred_std'],
                val_stats['pred_std'],
                t1 - t0,
            )
        else:
            logger.info(
                'Epoch %d/%d - lr=%.2e train_loss=%.4f train_mse=%.4f val_mse=%.4f val_loss=%.4f '
                'train_pearson=%.4f val_pearson=%.4f train_pearson_lag=%.4f val_pearson_lag=%.4f '
                'train_pearson_lag_hard=%.4f val_pearson_lag_hard=%.4f train_hard_lag_mode=%d val_hard_lag_mode=%d '
                'train_soft_lag_mean=%.4f val_soft_lag_mean=%.4f train_soft_lag_median=%.4f val_soft_lag_median=%.4f '
                'train_pred_std=%.4f val_pred_std=%.4f time=%.1fs',
                epoch,
                args.epochs,
                current_lr,
                train_loss,
                train_mse,
                val_mse,
                val_loss,
                train_eval_pearson,
                val_pearson,
                train_eval_pearson_lag,
                val_pearson_lag,
                train_hard_pearson,
                val_hard_pearson,
                train_hard_lag_mode,
                val_hard_lag_mode,
                train_soft_lag_mean,
                val_soft_lag_mean,
                train_soft_lag_median,
                val_soft_lag_median,
                train_stats['pred_std'],
                val_stats['pred_std'],
                t1 - t0,
            )
        if train_stats['pred_std'] < 1e-3:
            logger.warning('Train predictions nearly constant (pred_std=%.6f). Consider lower weight_decay or disabling augment.', train_stats['pred_std'])
        if val_loader is not None and val_stats['pred_std'] < 1e-3:
            logger.warning('Val predictions nearly constant (pred_std=%.6f). Potential mean-collapse.', val_stats['pred_std'])

        # Keep large LR for the first warmup epochs, then cosine-decay for a fixed window.
        if epoch >= default_warmup_epochs and cosine_steps_done < default_cosine_epochs:
            scheduler.step()
            cosine_steps_done += 1
        # TensorBoard logs
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/pred_std', train_stats['pred_std'], epoch)
            writer.add_scalar('train/pearson_zero_lag', train_eval_pearson, epoch)
            writer.add_scalar('train/pearson_lag_comp', train_eval_pearson_lag, epoch)
            if val_loader is not None:
                writer.add_scalar('val/mse', val_mse, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/pred_std', val_stats['pred_std'], epoch)
                writer.add_scalar('val/pearson_zero_lag', val_pearson, epoch)
                writer.add_scalar('val/pearson_lag_comp', val_pearson_lag, epoch)
        ckpt = save_dir / f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt)
        # save best model only if we have a valid validation Pearson
        try:
            is_better = np.isfinite(val_pearson) and (val_pearson > best_val)
        except Exception:
            is_better = False
        if is_better:
            best_val = val_pearson
            patience_counter = 0  # Reset patience counter on improvement
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, save_dir / 'best.pt')
            logger.info('📈 Best model saved at epoch %d (val_pearson=%.4f)', epoch, val_pearson)
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                logger.warning(
                    '⏹️  Early stopping triggered: val_pearson did not improve for %d epochs. Best val_pearson=%.4f at epoch %d',
                    early_stopping_patience, best_val, epoch - patience_counter
                )
                break
        # record metrics
        epochs_done.append(epoch)
        train_losses.append(train_loss)
        train_maes.append(train_mse)
        val_maes.append(val_mse)
        train_pearsons.append(train_eval_pearson)
        val_pearsons.append(val_pearson)
        train_pred_stds.append(train_stats['pred_std'])
        val_pred_stds.append(val_stats['pred_std'])
        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                train_loss,
                train_mse,
                val_mse,
                val_loss,
                train_eval_pearson,
                val_pearson,
                train_eval_pearson_lag,
                val_pearson_lag,
                train_hard_pearson,
                val_hard_pearson,
                train_hard_lag_mode,
                val_hard_lag_mode,
                train_soft_lag_mean,
                val_soft_lag_mean,
                train_soft_lag_median,
                val_soft_lag_median,
                json.dumps(train_lag_stats.get('sample_100', []), ensure_ascii=True),
                json.dumps(val_lag_stats.get('sample_100', []), ensure_ascii=True),
                train_stats['pred_std'],
                val_stats['pred_std'],
            ])
        if epoch % 10 == 0:
            with open(lag_samples_csv, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    epoch,
                    json.dumps(train_lag_stats.get('sample_100', []), ensure_ascii=True),
                    json.dumps(val_lag_stats.get('sample_100', []), ensure_ascii=True),
                ])

        # try to save simple plots for quick visualization
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(8, 4))
            plt.plot(epochs_done, train_losses, label='train_loss')
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.title('Train loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / 'train_loss.png')
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(epochs_done, train_maes, label='train_mse')
            plt.plot(epochs_done, val_maes, label='val_mse')
            plt.xlabel('epoch')
            plt.ylabel('MSE (PPG)')
            plt.title('PPG Waveform MSE over epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / 'mse_epochs.png')
            plt.close()
        except Exception:
            logger.warning('Could not write training plots (matplotlib not available)')

        # Visualize attention heatmaps every epoch
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np

            model.eval()
            with torch.no_grad():
                # Randomly sample clips from different subjects for visualization
                if val_loader is not None and len(val_ds.files) > 0:
                    candidate_files = list(val_ds.files)
                    random.shuffle(candidate_files)

                    selected_files = []
                    used_subjects = set()
                    for fn in candidate_files:
                        subj = subject_from_name(fn)
                        if subj in used_subjects:
                            continue
                        selected_files.append(fn)
                        used_subjects.add(subj)
                        if len(selected_files) >= 4:
                            break

                    if len(selected_files) > 0:
                        clips_list = []
                        roi_types = []
                        vis_subjects = []
                        for fn in selected_files:
                            data = torch.load(fn, weights_only=False)
                            clips_list.append(data['clip'].float())
                            roi_types.append(data.get('roi_type', 'full'))
                            vis_subjects.append(subject_from_name(fn))

                        clips = torch.stack(clips_list, dim=0).to(device)
                        roi_map = _build_roi_map(clips, roi_types, device)

                        # Forward with attention
                        _out, attn_dict = model(clips, roi_map=roi_map, return_attention=True)

                        # Visualize g1 and g2
                        g1 = attn_dict['g1']  # (B, T, 1, H1, W1)
                        g2 = attn_dict['g2']  # (B, T, 1, H2, W2)

                        num_viz = g1.shape[0]

                        fig, axes = plt.subplots(num_viz, 4, figsize=(12, 3 * num_viz))
                        if num_viz == 1:
                            axes = axes.reshape(1, -1)

                        for bi in range(num_viz):
                            # First frame, first gate
                            ax = axes[bi, 0]
                            hm = g1[bi, 0, 0].cpu().numpy()
                            im = ax.imshow(hm, cmap='hot')
                            ax.set_title(f'{vis_subjects[bi]} G1 Frame 0')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046)

                            # Middle frame, first gate
                            ax = axes[bi, 1]
                            mid_t = g1.shape[1] // 2
                            hm = g1[bi, mid_t, 0].cpu().numpy()
                            im = ax.imshow(hm, cmap='hot')
                            ax.set_title(f'{vis_subjects[bi]} G1 Frame {mid_t}')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046)

                            # First frame, second gate
                            ax = axes[bi, 2]
                            hm = g2[bi, 0, 0].cpu().numpy()
                            im = ax.imshow(hm, cmap='hot')
                            ax.set_title(f'{vis_subjects[bi]} G2 Frame 0')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046)

                            # Middle frame, second gate
                            ax = axes[bi, 3]
                            mid_t = g2.shape[1] // 2
                            hm = g2[bi, mid_t, 0].cpu().numpy()
                            im = ax.imshow(hm, cmap='hot')
                            ax.set_title(f'{vis_subjects[bi]} G2 Frame {mid_t}')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046)

                        plt.suptitle(f'Epoch {epoch} - Attention Heatmaps (different subjects)', fontsize=14)
                        plt.tight_layout()
                        plt.savefig(save_dir / f'attention_epoch_{epoch:03d}.png', dpi=80)
                        plt.close()
                        logger.info(f'Saved attention heatmap visualization to {save_dir / f"attention_epoch_{epoch:03d}.png"}')
        except Exception as e:
            logger.warning(f'Could not save attention heatmaps: {e}')


if __name__ == '__main__':
    main()

