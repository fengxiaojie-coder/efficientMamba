from __future__ import annotations

import numpy as np


class BaseSignalNormalizationMethod:
    name = 'base'

    def __init__(self, config):
        self.config = config

    def process(self, ppg: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentitySignalNormalizationMethod(BaseSignalNormalizationMethod):
    name = 'none'

    def process(self, ppg: np.ndarray) -> np.ndarray:
        return np.asarray(ppg, dtype=np.float32).squeeze()


class ZScoreSignalNormalizationMethod(BaseSignalNormalizationMethod):
    name = 'zscore'

    def process(self, ppg: np.ndarray) -> np.ndarray:
        sig = np.asarray(ppg, dtype=np.float32).squeeze()
        if sig.size == 0:
            return np.zeros((0,), dtype=np.float32)
        mean = float(np.mean(sig))
        std = float(np.std(sig))
        if std < 1e-6:
            return (sig - mean).astype(np.float32)
        return ((sig - mean) / std).astype(np.float32)


class MinMaxSignalNormalizationMethod(BaseSignalNormalizationMethod):
    name = 'minmax'

    def process(self, ppg: np.ndarray) -> np.ndarray:
        sig = np.asarray(ppg, dtype=np.float32).squeeze()
        if sig.size == 0:
            return np.zeros((0,), dtype=np.float32)
        min_val = float(np.min(sig))
        max_val = float(np.max(sig))
        if max_val - min_val < 1e-6:
            return np.zeros_like(sig, dtype=np.float32)
        return ((sig - min_val) / (max_val - min_val)).astype(np.float32)


class RobustZSignalNormalizationMethod(BaseSignalNormalizationMethod):
    name = 'robust_z'

    def process(self, ppg: np.ndarray) -> np.ndarray:
        sig = np.asarray(ppg, dtype=np.float32).squeeze()
        if sig.size == 0:
            return np.zeros((0,), dtype=np.float32)
        median = float(np.median(sig))
        mad = float(np.median(np.abs(sig - median)))
        if mad < 1e-6:
            return (sig - median).astype(np.float32)
        return ((sig - median) / (1.4826 * mad)).astype(np.float32)