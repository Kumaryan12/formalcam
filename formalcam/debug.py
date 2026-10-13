"""Preview-only diagnostics. Never applied to the virtual-camera feed."""
import cv2
import numpy as np

from .masks import composite_garment
from .pose import BodyPose


def debug_preview(image: np.ndarray, pose: BodyPose | None,
                  mask: np.ndarray | None, landmarks: bool, show_mask: bool) -> np.ndarray:
    preview = image.copy()
    if show_mask and mask is not None:
        green = np.full_like(preview, (45, 205, 65))
        preview = composite_garment(preview, green, mask*.45)
    if landmarks and pose is not None:
        edges = (('left_shoulder','right_shoulder'), ('left_shoulder','left_elbow'),
                 ('left_elbow','left_wrist'), ('right_shoulder','right_elbow'),
                 ('right_elbow','right_wrist'), ('left_shoulder','left_hip'),
                 ('right_shoulder','right_hip'), ('left_hip','right_hip'))
        for a_name, b_name in edges:
            a, b = getattr(pose, a_name), getattr(pose, b_name)
            if a is not None and b is not None:
                cv2.line(preview, (round(a.x),round(a.y)), (round(b.x),round(b.y)), (20,235,255), 2, cv2.LINE_AA)
        for point in vars(pose).values():
            if point is not None:
                cv2.circle(preview, (round(point.x),round(point.y)), 4, (255,240,20), -1, cv2.LINE_AA)
    return preview
