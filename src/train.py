"""Training script for EfficientMamba.

Usage (quick run):
    python -m src.train --clips_dir data/ubfc_clips --epochs 2 --batch_size 4 --max_samples 200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import time
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import torch
from torch import nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

from src.datasets.ubfc_dataset import UBFCClipDataset
from src.models.efficientphys_mamba import EfficientPhysMambaRegressor

# optional fast progress bars; fallback to no-op if tqdm not installed
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
import csv


def train_epoch(model, loader, optimizer, loss_fn, device, desc: str | None = None):
    model.train()
    total_loss = 0.0
    count = 0
    iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        # support datasets that may return ROI metadata: (clips, hrs[, roi_type, roi_bbox])
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            clips = batch[0]
            hrs = batch[1]
            roi_types = batch[2] if len(batch) > 2 else None
        else:
            clips, hrs = batch
            roi_types = None

        clips = clips.to(device)
        hrs = hrs.to(device).unsqueeze(1)

        # build a simple centered Gaussian ROI map per-sample when explicit ROI was used
        roi_map = None
        if roi_types is not None:
            try:
                B = clips.size(0)
                H = clips.size(-2)
                W = clips.size(-1)
                mask = torch.ones((B, 1, H, W), device=device)
                for i, rt in enumerate(roi_types):
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

        optimizer.zero_grad()
        preds = model(clips, roi_map=roi_map) if roi_map is not None else model(clips)
        loss = loss_fn(preds, hrs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        count += clips.size(0)
    return total_loss / max(1, count)


def eval_model(model, loader, device, desc: str | None = None):
    model.eval()
    import numpy as np
    preds_all = []
    hrs_all = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                clips = batch[0]
                hrs = batch[1]
                roi_types = batch[2] if len(batch) > 2 else None
            else:
                clips, hrs = batch
                roi_types = None

            clips = clips.to(device)

            roi_map = None
            if roi_types is not None:
                try:
                    B = clips.size(0)
                    H = clips.size(-2)
                    W = clips.size(-1)
                    mask = torch.ones((B, 1, H, W), device=device)
                    for i, rt in enumerate(roi_types):
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

            out = model(clips, roi_map=roi_map) if roi_map is not None else model(clips)
            preds_all.append(out.cpu().numpy())
            hrs_all.append(hrs.numpy())
    preds = np.vstack(preds_all).ravel()
    hrs = np.hstack(hrs_all).ravel()
    mae = float(np.mean(np.abs(preds - hrs)))
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_dir', default='data/ubfc_clips')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay (L2 regularization)')
    parser.add_argument('--augment', action='store_true', help='Enable simple data augmentations on the training set')
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
    args = parser.parse_args()

    # configure logging early so modules emit consistent messages
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    ds = UBFCClipDataset(args.clips_dir)
    # Subject-aware selection
    def subject_from_name(fn: Path) -> str:
        name = fn.name
        return name.split('_clip_', 1)[0] if '_clip_' in name else name

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
        raise SystemExit('No clips found. Run preprocessing first.')
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
    train_ds = UBFCClipDataset(args.clips_dir, augment=args.augment)
    train_ds.files = train_files
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if len(val_files) > 0:
        val_ds = UBFCClipDataset(args.clips_dir, augment=False)
        val_ds.files = val_files
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        val_loader = None

    model = EfficientPhysMambaRegressor(in_channels=6)
    model.to(device)

    optimizer = torch_optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'runs')) if SummaryWriter is not None else None

    best_val = float('inf')
    epochs_done = []
    train_losses = []
    train_maes = []
    val_maes = []
    metrics_csv = save_dir / 'training_metrics.csv'
    # write header
    with open(metrics_csv, 'w', newline='') as _f:
        w = csv.writer(_f)
        w.writerow(['epoch', 'train_loss', 'train_mae', 'val_mae'])
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        # compute train MAE for monitoring
        train_mae = eval_model(model, train_loader, device)
        # compute val MAE if validation set exists
        if val_loader is not None:
            val_mae = eval_model(model, val_loader, device)
        else:
            val_mae = float('nan')
        t1 = time.time()
        logger.info('Epoch %d/%d - train_loss=%.4f train_mae=%.4f val_mae=%.4f time=%.1fs', epoch, args.epochs, train_loss, train_mae, val_mae, t1-t0)
        # TensorBoard logs
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            if val_loader is not None:
                writer.add_scalar('val/mae', val_mae, epoch)
        ckpt = save_dir / f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt)
        # save best model only if we have a valid validation MAE
        try:
            is_better = val_mae < best_val
        except Exception:
            is_better = False
        if is_better:
            best_val = val_mae
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, save_dir / 'best.pt')
        # record metrics
        epochs_done.append(epoch)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_mae, val_mae])

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
            plt.plot(epochs_done, train_maes, label='train_mae')
            plt.plot(epochs_done, val_maes, label='val_mae')
            plt.xlabel('epoch')
            plt.ylabel('MAE (bpm)')
            plt.title('MAE over epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / 'mae_epochs.png')
            plt.close()
        except Exception:
            logger.warning('Could not write training plots (matplotlib not available)')


if __name__ == '__main__':
    main()

