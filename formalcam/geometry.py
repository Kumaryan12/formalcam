"""Shoulder-anchored 2D torso deformation, with bounded optional hip estimates."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .pose import BodyPose, temporal_alpha


@dataclass
class TorsoGeometry:
    center: np.ndarray
    across: np.ndarray
    down: np.ndarray
    shoulder_width: float
    width_scale: float
    lean: float = 0.
    waist_ratio: float = 1.
    length_ratio: float = 1.05

    def points(self, coordinates: Sequence[tuple[float, float]]) -> np.ndarray:
        uv = np.asarray(coordinates, np.float32)
        u, v = uv[:, 0], uv[:, 1]
        weight = np.clip(v / 1.05, 0., 1.)
        across = self.width_scale * u * (1. + weight * (self.waist_ratio - 1.))
        across += weight * self.lean
        depth = np.where(v > 0., v * self.length_ratio / 1.05, v)
        points = self.center + self.shoulder_width * (
            across[:, None] * self.across + depth[:, None] * self.down)
        return np.rint(points).astype(np.int32)


class TorsoFitter:
    """Smooth valid hip shape changes; always release the garment on pose loss."""
    def __init__(self) -> None:
        self._shape: np.ndarray | None = None

    def reset(self) -> None:
        self._shape = None

    def fit(self, pose: BodyPose | None, frame_shape: tuple[int, int],
            width_scale: float = 1., vertical_offset: float = 0.,
            elapsed_s: float = 1/30) -> TorsoGeometry | None:
        if pose is None:
            self.reset()
            return None
        a, b = sorted((pose.left_shoulder, pose.right_shoulder), key=lambda point: point.x)
        delta = np.array([b.x-a.x, b.y-a.y], np.float32)
        width = float(np.linalg.norm(delta))
        if not math.isfinite(width) or width < 35.:
            self.reset()
            return None
        across = delta / width
        down = np.array([-across[1], across[0]], np.float32)
        center = np.array([(a.x+b.x)*.5, (a.y+b.y)*.5], np.float32)
        target = np.array([0., 1., 1.05], np.float32)
        h, w = frame_shape
        hips = pose.left_hip, pose.right_hip
        if all(p is not None and p.confidence >= .4 and
               0 <= p.x < w and 0 <= p.y < h for p in hips):
            left, right = hips
            hip_center = np.array([(left.x+right.x)*.5, (left.y+right.y)*.5])
            offset = (hip_center-center) / width
            length = float(offset @ down)
            waist = abs(float(np.array([right.x-left.x, right.y-left.y]) @ across)) / width
            lean = float(offset @ across)
            if .5 <= length <= 2.2 and .25 <= waist <= 1.3 and abs(lean) <= .6:
                target = np.array([np.clip(lean, -.4, .4), np.clip(waist, .55, 1.15),
                                   np.clip(length, .7, 1.65)], np.float32)
        alpha = temporal_alpha(.35, elapsed_s)
        self._shape = target if self._shape is None else self._shape + alpha*(target-self._shape)
        shifted = center + width * vertical_offset * down
        return TorsoGeometry(shifted, across, down, width, width_scale, *map(float, self._shape))
