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
        self.output_frames = 0
        self.output_fps = 0.0
        self.capture_fps = 0.0
        self.segmentation_results = 0
        self.totals = dict.fromkeys(('capture', 'pose', 'segmentation', 'render', 'composite', 'send', 'display'), 0.0)
        self.writer = csv.writer(stream) if stream is not None else None
        if self.writer:
            self.writer.writerow(('frame', 'elapsed_s', 'pose_detected', 'fps',
                                  *[f'{name}_ms' for name in self.totals], 'output_sent',
                                  'output_fps', 'capture_fps', 'seg_inference_ms', 'seg_fresh', 'seg_age_ms'))

    def update(self, now: float, detected: bool, timings: dict[str, float], *,
               output_sent: bool = False, capture_frames: int | None = None,
               seg_inference_ms: float = 0., seg_fresh: bool = False,
               seg_age_ms: float | None = None) -> None:
        self.frames += 1
        self.pose_frames += int(detected)
        elapsed = max(now - self.start, 1e-9)
        self.fps = self.frames / elapsed
        self.output_frames += int(output_sent)
        self.output_fps = self.output_frames / elapsed
        self.capture_fps = (self.frames if capture_frames is None else capture_frames) / elapsed
        self.segmentation_results += int(seg_fresh)
        for name in self.totals:
            self.totals[name] += timings.get(name, 0.)
        if self.writer:
            self.writer.writerow((self.frames, f'{elapsed:.6f}', int(detected),
                                  f'{self.fps:.3f}',
                                  *[f'{timings.get(name, 0.) * 1000:.3f}' for name in self.totals],
                                  int(output_sent), f'{self.output_fps:.3f}', f'{self.capture_fps:.3f}',
                                  f'{seg_inference_ms:.3f}', int(seg_fresh),
                                  '' if seg_age_ms is None else f'{seg_age_ms:.3f}'))

    def summary(self) -> str:
        count = max(1, self.frames)
        stages = ' '.join(f'{name}={value * 1000 / count:.1f}ms'
                          for name, value in self.totals.items())
        return (f'{self.frames} frames | {self.fps:.1f} fps | '
                f'pose {self.pose_frames}/{self.frames} | capture {self.capture_fps:.1f} fps '
                f'output {self.output_fps:.1f} fps | seg updates {self.segmentation_results} | {stages}')


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
