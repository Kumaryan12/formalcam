"""Local pose -> garment -> protected composite; the sender gets clean BGR only."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from .masks import composite_garment, protect_landmarks
from .pose import BodyPose, PoseSmoother, from_mediapipe
from .render import GarmentRenderer, RenderOptions
from .segmentation import MaskUpdate, SegmentationEngine


@dataclass
class ProcessedFrame:
    image: np.ndarray
    pose: BodyPose | None
    mask: MaskUpdate
    timings: dict[str, float]
    segmentation_error: str | None = None


class FrameProcessor:
    def __init__(self, landmarker: Any, segmentation: SegmentationEngine):
        self.landmarker, self.segmentation = landmarker, segmentation
        self.smoother, self.renderer = PoseSmoother(), GarmentRenderer()

    def process(self, image: np.ndarray, timestamp_ms: int, source_time: float,
                elapsed_s: float, opts: RenderOptions, overlay: bool,
                segmentation_enabled: bool) -> ProcessedFrame:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Camera input must be HxWx3 uint8 BGR')
        import mediapipe as mp
        start = time.monotonic()
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        h, w = image.shape[:2]
        pose = from_mediapipe(result.pose_landmarks[0], w, h) if result.pose_landmarks else None
        pose = self.smoother.update(pose, elapsed_s)
        posed = time.monotonic()
        mask = MaskUpdate(None)
        if pose is not None and overlay and segmentation_enabled:
            mask = self.segmentation.update(image, timestamp_ms, source_time, posed)
            if mask.allowance is not None:
                mask.allowance = protect_landmarks(mask.allowance, pose, (h,w))
        else:
            self.segmentation.reset()
        segmented = time.monotonic()
        can_paint = pose is not None and overlay and (not segmentation_enabled or mask.allowance is not None)
        if can_paint:
            rendered = self.renderer.render(image, pose, opts, elapsed_s)
        else:
            self.renderer.reset()
            rendered = image.copy()
        painted = time.monotonic()
        processed = composite_garment(image, rendered, mask.allowance) if can_paint and segmentation_enabled else rendered
        composited = time.monotonic()
        timings = dict(pose=posed-start, segmentation=segmented-posed,
                       render=painted-segmented, composite=composited-painted)
        return ProcessedFrame(processed, pose, mask, timings,
                              self.segmentation.error if segmentation_enabled else None)
