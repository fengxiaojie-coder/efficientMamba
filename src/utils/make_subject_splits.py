"""Create subject-disjoint train/val file lists for UBFC-style .pt clips.

Usage:
    python -m src.utils.make_subject_splits --clips_dir data/ubfc_clips \
        --out_dir splits --val_frac 0.1 --seed 0

The script writes `train.txt` and `val.txt` (one relative filepath per line)
into `out_dir`. Use these lists to ensure evaluation uses held-out clips.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import random
from typing import List


def subject_from_name(fn: Path) -> str:
    name = fn.name
    return name.split('_clip_', 1)[0] if '_clip_' in name else name


def make_subject_disjoint(clips_dir: Path, val_frac: float = 0.1, seed: int = 0) -> tuple[List[Path], List[Path]]:
    files = sorted(clips_dir.glob('*.pt'))
    subj_map: dict[str, list[Path]] = {}
    for fn in files:
        s = subject_from_name(fn)
        subj_map.setdefault(s, []).append(fn)

    subjects = sorted(subj_map.keys())
    random.seed(seed)
    random.shuffle(subjects)
    val_n = max(1, int(len(subjects) * val_frac))
    val_subjects = set(subjects[:val_n])
    train_files: List[Path] = []
    val_files: List[Path] = []
    for s in subjects:
        if s in val_subjects:
            val_files.extend(sorted(subj_map[s]))
        else:
            train_files.extend(sorted(subj_map[s]))
    return train_files, val_files


def write_list(paths: List[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(str(p.name) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--clips_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, default=Path('splits'))
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_files, val_files = make_subject_disjoint(args.clips_dir, val_frac=args.val_frac, seed=args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_list(train_files, out_dir / 'train.txt')
    write_list(val_files, out_dir / 'val.txt')
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Wrote %d train and %d val files to %s', len(train_files), len(val_files), out_dir)


if __name__ == '__main__':
    main()
