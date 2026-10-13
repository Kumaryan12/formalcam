"""Conservative clothing masks, low-resolution propagation, and BGR compositing."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .pose import BodyPose


def clothing_allowance(masks: Sequence[np.ndarray], labels: Sequence[str]) -> np.ndarray:
    normalized = [label.strip().lower().replace('_', '-') for label in labels]
    expected = {'background', 'hair', 'body-skin', 'face-skin', 'clothes', 'others'}
    if len(masks) != 6 or set(normalized) != expected or len(normalized) != 6:
        raise ValueError('Expected the six SelfieMulticlass category labels')
    arrays = [np.asarray(mask, np.float32) for mask in masks]
    shape = arrays[0].shape
    if len(shape) != 2 or any(a.shape != shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Segmentation confidence maps must be finite and have equal 2D shapes')
    clothes = np.clip(arrays[normalized.index('clothes')], 0., 1.)
    protected = np.clip(sum(a for i, a in enumerate(arrays) if normalized[i] != 'clothes'), 0., 1.)
    allow = np.clip((clothes-.5)/.35, 0., 1.)
    allow = cv2.GaussianBlur(allow, (3, 3), .6)
    # Never soften high-confidence face/hair/skin/background into eligible clothing.
    allow[(protected >= clothes) | (clothes <= .5)] = 0.
    return allow


def propagate_mask(mask: np.ndarray, previous: np.ndarray,
                   current: np.ndarray) -> np.ndarray | None:
    if mask.shape != current.shape or previous.shape != current.shape:
        return None
    if np.mean(cv2.absdiff(previous, current)) > 45.:
        return None  # Scene change: don't retain a former subject's mask.
    flow = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 2, 9, 2, 5, 1.1, 0)
    yy, xx = np.indices(current.shape, np.float32)
    map_x, map_y = xx+flow[..., 0], yy+flow[..., 1]
    warped = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    prediction = cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Newly appearing foreground (e.g. a hand) should show original pixels while
    # awaiting a fresh inference. Optical flow is only an approximation of motion.
    warped[cv2.absdiff(prediction, current) > 30] = 0.
    return warped


def protect_landmarks(allowance: np.ndarray, pose: BodyPose,
                      frame_shape: tuple[int, int]) -> np.ndarray:
    """Additional conservative head/hand guards, not anatomical segmentation."""
    h, w = frame_shape
    mh, mw = allowance.shape
    sx, sy = mw/w, mh/h
    width = float(np.hypot(pose.left_shoulder.x-pose.right_shoulder.x,
                           pose.left_shoulder.y-pose.right_shoulder.y))
    mask = allowance.copy()
    if pose.nose is not None and pose.nose.confidence >= .6:
        point = pose.nose
        center = (round(point.x*sx), round((point.y-.08*width)*sy))
        axes = (max(1,round(width*.22*sx)), max(1,round(width*.38*sy)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 0., -1)
    for wrist in (pose.left_wrist, pose.right_wrist):
        if wrist is not None and wrist.confidence >= .6:
            cv2.ellipse(mask, (round(wrist.x*sx),round(wrist.y*sy)),
                        (max(1,round(width*.08*sx)), max(1,round(width*.1*sy))),
                        0, 0, 360, 0., -1)
    return mask


def composite_garment(original: np.ndarray, rendered: np.ndarray,
                      allowance: np.ndarray) -> np.ndarray:
    if original.shape != rendered.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Original and rendered frames must have matching HxWx3 BGR shapes')
    if original.dtype != np.uint8 or rendered.dtype != np.uint8:
        raise ValueError('Frames must use uint8 BGR')
    if allowance.ndim != 2 or not np.isfinite(allowance).all():
        raise ValueError('Clothing allowance must be a finite 2D mask')
    h, w = original.shape[:2]
    alpha = np.clip(cv2.resize(allowance.astype(np.float32), (w,h), interpolation=cv2.INTER_LINEAR), 0., 1.)
    output = original.copy()
    bounds = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if bounds is None:
        return output
    x, y, bw, bh = cv2.boundingRect(bounds)
    region = np.s_[y:y+bh, x:x+bw]
    base = original[region].astype(np.float32)
    output[region] = (base + (rendered[region].astype(np.float32)-base)*alpha[region][..., None]).astype(np.uint8)
    return output
