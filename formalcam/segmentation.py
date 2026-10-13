"""Optional local multiclass segmentation, one in-flight job, bounded mask age."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .masks import clothing_allowance, propagate_mask
from .privacy import prepare_pose_runtime

MODEL = Path(__file__).resolve().parent.parent / 'models' / 'selfie_multiclass_256x256.tflite'


@dataclass
class MaskUpdate:
    allowance: np.ndarray | None
    inference_ms: float = 0.
    fresh: bool = False
    age_ms: float | None = None


class SegmentationEngine:
    def __init__(self, model_path: Path = MODEL, interval_s: float = .1,
                 max_age_s: float = .25,
                 predictor: Callable[[np.ndarray, int], np.ndarray] | None = None):
        if interval_s <= 0 or max_age_s <= 0:
            raise ValueError('Segmentation cadence and maximum mask age must be positive')
        self.model_path, self.interval, self.max_age = model_path, interval_s, max_age_s
        self.error: str | None = None
        self._predictor = predictor
        self._model = None
        self._pool: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._generation = 0
        self._mask: np.ndarray | None = None
        self._gray: np.ndarray | None = None
        self._source_time: float | None = None
        self._last_scheduled = -float('inf')
        if predictor is None and not model_path.is_file():
            self.error = f'Missing segmentation model: {model_path}; run python scripts/download_model.py --segmentation'

    def reset(self) -> None:
        self._mask, self._gray, self._source_time = None, None, None
        self._generation += 1  # An old result must not survive loss of tracking.

    def _predict(self, bgr: np.ndarray, timestamp: int) -> np.ndarray:
        if self._predictor is not None:
            return self._predictor(bgr, timestamp)
        prepare_pose_runtime()
        import mediapipe as mp
        if self._model is None:
            options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_category_mask=False, output_confidence_masks=True)
            self._model = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        start = time.monotonic()
        result = self._model.segment_for_video(image, timestamp)
        self._last_inference_ms = (time.monotonic()-start)*1000
        return clothing_allowance([mask.numpy_view() for mask in result.confidence_masks], self._model.labels)

    def _job(self, bgr: np.ndarray, gray: np.ndarray, timestamp: int,
             source_time: float, generation: int) -> tuple:
        start = time.monotonic()
        allowance = self._predict(bgr, timestamp)
        duration = (time.monotonic()-start)*1000 if self._predictor is not None else self._last_inference_ms
        allowance = cv2.resize(allowance, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(allowance).all():
            raise ValueError('Segmenter returned a nonfinite mask')
        return allowance, gray, source_time, generation, duration

    def update(self, frame: np.ndarray, timestamp_ms: int, source_time: float,
               now: float) -> MaskUpdate:
        if self.error:
            return MaskUpdate(None)
        # Flow uses only 160-wide luminance; the model receives a 256-wide frame.
        height = max(8, round(160*frame.shape[0]/frame.shape[1]))
        gray = cv2.cvtColor(cv2.resize(frame, (160,height)), cv2.COLOR_BGR2GRAY)
        if self._mask is not None and self._gray is not None:
            self._mask = propagate_mask(self._mask, self._gray, gray)
        self._gray = gray
        inference_ms, fresh = 0., False
        if self._future is not None and self._future.done():
            try:
                mask, captured_gray, stamp, generation, inference_ms = self._future.result()
                if generation == self._generation and now-stamp <= self.max_age:
                    mask = propagate_mask(mask, captured_gray, gray)
                    if mask is not None:
                        if self._mask is not None:
                            protected = mask <= 0.
                            mask = .7*mask + .3*self._mask
                            # A freshly protected region takes precedence over history.
                            mask[protected] = 0.
                        self._mask, self._source_time, fresh = mask, stamp, True
            except Exception as exc:
                self.error = f'Segmentation disabled: {exc}'
                self.reset()
            self._future = None
        if not self.error and self._future is None and now-self._last_scheduled >= self.interval:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='formalcam-segmentation')
            height_rgb = max(8, round(256*frame.shape[0]/frame.shape[1]))
            small = cv2.resize(frame, (256,height_rgb))
            self._future = self._pool.submit(self._job, small, gray.copy(), timestamp_ms, source_time, self._generation)
            self._last_scheduled = now
        age = None if self._source_time is None else max(0., now-self._source_time)
        if age is None or age > self.max_age:
            self._mask = None
        return MaskUpdate(self._mask, inference_ms, fresh, None if age is None else age*1000)

    def _close_model(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.submit(self._close_model).result()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self.reset()
