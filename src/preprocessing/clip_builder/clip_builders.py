from __future__ import annotations

import numpy as np


class BaseClipBuilder:
    name = 'base'

    def __init__(self, config):
        self.config = config

    def process(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _windows(self, frames: np.ndarray):
        num_frames, height, width, _channels = frames.shape
        for start in range(0, max(1, num_frames - self.config.clip_len + 1), self.config.stride):
            yield start, frames[start:start + self.config.clip_len].astype(np.float32) / 255.0, height, width


class RawPlusDiffClipBuilder(BaseClipBuilder):
    name = 'raw_plus_diff'

    def process(self, frames: np.ndarray) -> np.ndarray:
        clips = []
        height = width = 0
        for _start, window, height, width in self._windows(frames):
            diffs = np.zeros_like(window)
            diffs[1:] = window[1:] - window[:-1]
            concat = np.concatenate([diffs, window], axis=-1)
            clips.append(np.transpose(concat, (0, 3, 1, 2)))
        if len(clips) == 0:
            return np.empty((0, self.config.clip_len, 6, height, width), dtype=np.float32)
        return np.stack(clips, axis=0)


class RawOnlyClipBuilder(BaseClipBuilder):
    name = 'raw_only'

    def process(self, frames: np.ndarray) -> np.ndarray:
        clips = []
        height = width = 0
        for _start, window, height, width in self._windows(frames):
            raw = np.transpose(window, (0, 3, 1, 2))
            clips.append(raw)
        if len(clips) == 0:
            return np.empty((0, self.config.clip_len, 3, height, width), dtype=np.float32)
        return np.stack(clips, axis=0)


class DiffOnlyClipBuilder(BaseClipBuilder):
    name = 'diff_only'

    def process(self, frames: np.ndarray) -> np.ndarray:
        clips = []
        height = width = 0
        for _start, window, height, width in self._windows(frames):
            diffs = np.zeros_like(window)
            diffs[1:] = window[1:] - window[:-1]
            clips.append(np.transpose(diffs, (0, 3, 1, 2)))
        if len(clips) == 0:
            return np.empty((0, self.config.clip_len, 3, height, width), dtype=np.float32)
        return np.stack(clips, axis=0)


class MeanCenteredClipBuilder(BaseClipBuilder):
    name = 'mean_centered'

    def process(self, frames: np.ndarray) -> np.ndarray:
        clips = []
        height = width = 0
        for _start, window, height, width in self._windows(frames):
            centered = window - np.mean(window, axis=0, keepdims=True)
            clips.append(np.transpose(centered, (0, 3, 1, 2)))
        if len(clips) == 0:
            return np.empty((0, self.config.clip_len, 3, height, width), dtype=np.float32)
        return np.stack(clips, axis=0)