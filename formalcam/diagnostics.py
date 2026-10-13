"""Local numeric timings and explicitly requested still images; no networking."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np


class FrameStats:
    def __init__(self, start: float, stream: TextIO | None = None):
        self.start = start
        self.frames = 0
        self.pose_frames = 0
        self.fps = 0.0
        self.totals = dict.fromkeys(('capture', 'pose', 'render', 'send', 'display'), 0.0)
        self.writer = csv.writer(stream) if stream is not None else None
        if self.writer:
            self.writer.writerow(('frame', 'elapsed_s', 'pose_detected', 'fps',
                                  *[f'{name}_ms' for name in self.totals]))

    def update(self, now: float, detected: bool, timings: dict[str, float]) -> None:
        self.frames += 1
        self.pose_frames += int(detected)
        elapsed = max(now - self.start, 1e-9)
        self.fps = self.frames / elapsed
        for name in self.totals:
            self.totals[name] += timings[name]
        if self.writer:
            self.writer.writerow((self.frames, f'{elapsed:.6f}', int(detected),
                                  f'{self.fps:.3f}',
                                  *[f'{timings[name] * 1000:.3f}' for name in self.totals]))

    def summary(self) -> str:
        count = max(1, self.frames)
        stages = ' '.join(f'{name}={value * 1000 / count:.1f}ms'
                          for name, value in self.totals.items())
        return (f'{self.frames} frames | {self.fps:.1f} fps | '
                f'pose {self.pose_frames}/{self.frames} | {stages}')


def save_snapshot(directory: Path, frame_number: int, original: np.ndarray,
                  preview: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Unique filenames prevent replacing a previous validation capture.
    import time
    prefix = f'{time.time_ns()}-{frame_number:06d}'
    for label, image in (('original', original), ('preview', preview)):
        path = directory / f'{prefix}-{label}.png'
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Could not write snapshot: {path}')
        print('Saved locally:', path, flush=True)
