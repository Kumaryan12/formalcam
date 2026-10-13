"""Minimal MediaPipe Tasks pose adapter with smoothing across webcam frames."""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass
class Point:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class BodyPose:
    left_shoulder: Point
    right_shoulder: Point
    left_elbow: Point | None = None
    right_elbow: Point | None = None
    left_wrist: Point | None = None
    right_wrist: Point | None = None
    left_hip: Point | None = None
    right_hip: Point | None = None
    nose: Point | None = None


def temporal_alpha(alpha: float, elapsed_s: float) -> float:
    """Preserve the nominal 30-FPS EMA response when frame cadence changes."""
    return 1.0 - (1.0 - alpha) ** (min(.2, max(0., elapsed_s)) * 30.)


def _point(landmarks: Sequence, index: int, w: int, h: int) -> Point | None:
    if index >= len(landmarks):
        return None
    p = landmarks[index]
    scores = [getattr(p, name, 1.0) for name in ('visibility', 'presence')]
    scores = [1.0 if value is None else float(value) for value in scores]
    x, y = float(p.x), float(p.y)
    if not all(math.isfinite(value) for value in (*scores, x, y)):
        return None
    confidence = min(scores)
    if confidence < .4:
        return None
    return Point(x * w, y * h, confidence)


def from_mediapipe(landmarks: Sequence, w: int, h: int) -> BodyPose | None:
    # After flipping the camera image, MediaPipe's left/right ANATOMICAL
    # labels need not correspond to the user's screen-left / screen-right.
    # Rendering uses image-coordinate ordering, not anatomy.
    l = _point(landmarks, 11, w, h)
    r = _point(landmarks, 12, w, h)
    if not l or not r:
        return None
    sep = float(np.hypot(l.x-r.x, l.y-r.y))
    if not 35 <= sep <= w * .85:
        return None
    return BodyPose(l, r,
                    _point(landmarks, 13, w, h),
                    _point(landmarks, 14, w, h),
                    _point(landmarks, 15, w, h),
                    _point(landmarks, 16, w, h),
                    _point(landmarks, 23, w, h),
                    _point(landmarks, 24, w, h),
                    _point(landmarks, 0, w, h))


class PoseSmoother:
    """Exponential smoothing to reduce jitter without storing video."""
    def __init__(self, alpha: float = 0.42):
        if not 0. <= alpha <= 1.:
            raise ValueError('Smoothing alpha must be between 0 and 1')
        self.alpha = alpha
        self.prev: BodyPose | None = None

    def update(self, pose: BodyPose | None, elapsed_s: float = 1/30) -> BodyPose | None:
        if pose is None:
            self.prev = None
            return None
        if self.prev is None:
            self.prev = pose
            return pose

        def mix(a: Point | None, b: Point | None) -> Point | None:
            if a is None or b is None:
                return b
            # Trust weaker but still valid observations less, without holding a lost pose.
            t = temporal_alpha(self.alpha * min(1., max(.5, b.confidence)), elapsed_s)
            return Point(a.x * (1-t) + b.x*t, a.y*(1-t)+b.y*t, b.confidence)

        p = self.prev
        self.prev = BodyPose(
            mix(p.left_shoulder, pose.left_shoulder),
            mix(p.right_shoulder, pose.right_shoulder),
            mix(p.left_elbow, pose.left_elbow),
            mix(p.right_elbow, pose.right_elbow),
            mix(p.left_wrist, pose.left_wrist),
            mix(p.right_wrist, pose.right_wrist),
            mix(p.left_hip, pose.left_hip),
            mix(p.right_hip, pose.right_hip),
            mix(p.nose, pose.nose),
        )
        return self.prev
