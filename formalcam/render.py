"""2D geometric blazer proof of concept, not photorealistic virtual try-on.

Only NumPy and OpenCV are required here; tests can run without webcam or MediaPipe.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .pose import BodyPose, temporal_alpha
from .geometry import TorsoFitter

# BGR colors, intentionally matte and simple (lighting is an approximation).
PALETTES = {
    'navy': (104, 65, 38),
    'charcoal': (65, 62, 60),
    'black': (33, 32, 30),
}


@dataclass
class RenderOptions:
    outfit: str = 'navy'
    lighting: bool = True
    shadows: bool = True
    restore_forearms: bool = False
    width_scale: float = 1.0
    vertical_offset: float = 0.0  # Shoulder-width units, positive moves down.


class GarmentRenderer:
    def __init__(self):
        self._brightness: float | None = None
        self._direction: float = 0.0
        self._fitter = TorsoFitter()

    def reset(self) -> None:
        self._brightness, self._direction = None, 0.0
        self._fitter.reset()

    def _estimate_light(self, image: np.ndarray, cx: float, sy: float,
                        shoulder_width: float, elapsed_s: float = 1/30) -> tuple[float, float]:
        h, w = image.shape[:2]
        # Face-adjacent region above shoulders; imperfect when face isn't centered.
        x0 = int(np.clip(cx - shoulder_width*.29, 0, w-1))
        x1 = int(np.clip(cx + shoulder_width*.29, x0+1, w))
        y0 = int(np.clip(sy - shoulder_width*.71, 0, h-1))
        y1 = int(np.clip(sy - shoulder_width*.15, y0+1, h))
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return 1.0, 0.0
        lightness = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Target maps roughly 110/255 face luminance to normal material colors.
        factor = float(np.clip((lightness.mean()+18) / 130., .67, 1.30))
        split = max(1, lightness.shape[1]//2)
        # Positive direction brightens screen-right (the panel's side=+1).
        directional = 0.0 if lightness.shape[1] < 2 else float(np.clip(
            (lightness[:,split:].mean() - lightness[:,:split].mean()) / 95, -.22, .22))
        alpha = temporal_alpha(.1, elapsed_s)
        self._brightness = factor if self._brightness is None else (1-alpha)*self._brightness + alpha*factor
        self._direction = (1-alpha)*self._direction + alpha*directional
        return self._brightness, self._direction

    @staticmethod
    def _adjust(color: tuple[int,int,int], scale: float) -> tuple[int,int,int]:
        return tuple(int(np.clip(c * scale, 0, 255)) for c in color)

    @staticmethod
    def _blend_shape(image: np.ndarray, coords: np.ndarray,
                     color: tuple[int,int,int], opacity: float = .98,
                     blur: float = 1.3) -> None:
        """Alpha blend only inside polygon bounding box (much faster than full frame)."""
        if coords.shape[0] < 3:
            return
        h, w = image.shape[:2]
        x0 = max(0, int(coords[:, 0].min()) - 5)
        x1 = min(w, int(coords[:, 0].max()) + 6)
        y0 = max(0, int(coords[:, 1].min()) - 5)
        y1 = min(h, int(coords[:, 1].max()) + 6)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        shifted = coords - np.asarray([x0,y0], dtype=np.int32)
        cv2.fillPoly(mask, [shifted], 255, cv2.LINE_AA)
        if blur:
            mask = cv2.GaussianBlur(mask, (5, 5), blur)
        a = (mask.astype(np.float32) * (opacity / 255.))[:, :, None]
        view = image[y0:y1, x0:x1]
        # A convex blend of uint8 colors is already inside [0,255].
        # Interpolate directly to avoid a full ROI clipping pass per polygon.
        view[:] = (view.astype(np.float32) +
                   (np.asarray(color,np.float32)[None,None,:] - view)*a).astype(np.uint8)

    @staticmethod
    def _segment_restore_mask(shape: tuple[int,int], pose: BodyPose, width: float) -> np.ndarray:
        """Approximate forearm occlusion; uses lines, not hand segmentation."""
        mask = np.zeros(shape, np.uint8)
        for elbow, wrist in ((pose.left_elbow, pose.left_wrist),
                              (pose.right_elbow, pose.right_wrist)):
            if elbow is None or wrist is None:
                continue
            # Extend slightly past wrist to include approximate hand area.
            p1 = np.array([elbow.x, elbow.y], dtype=np.float32)
            p2 = np.array([wrist.x, wrist.y], dtype=np.float32)
            p3 = p2 + (p2 - p1) * .38
            thick = max(9, int(width * .13))
            cv2.line(mask, tuple(p1.astype(int)), tuple(p3.astype(int)), 255, thick, cv2.LINE_AA)
        return cv2.GaussianBlur(mask, (9,9), 2)

    def render(self, frame: np.ndarray, pose: BodyPose | None,
               opts: RenderOptions | None = None, elapsed_s: float = 1/30) -> np.ndarray:
        opts = opts or RenderOptions()
        if pose is None:
            self.reset()
            return frame.copy()
        if opts.outfit not in PALETTES:
            raise ValueError(f'Unknown outfit: {opts.outfit}')
        if not .6 <= opts.width_scale <= 1.4 or not -.3 <= opts.vertical_offset <= .3:
            raise ValueError('Fit width must be 0.6..1.4 and offset -0.3..0.3')

        image = frame.copy()
        h,w = image.shape[:2]
        geometry = self._fitter.fit(pose, (h,w), opts.width_scale, opts.vertical_offset, elapsed_s)
        if geometry is None:
            return image
        measured_width = geometry.shoulder_width
        center = geometry.center
        width = measured_width * opts.width_scale

        def poly(points: list[tuple[float,float]]) -> np.ndarray:
            return geometry.points(points)

        light_center = center - measured_width * opts.vertical_offset * geometry.down
        scale,direction = self._estimate_light(frame, light_center[0], light_center[1], measured_width, elapsed_s) if opts.lighting else (1.0,0.0)
        base = PALETTES[opts.outfit]
        # Keep garment in view. A seated webcam often does not expose hips.
        extent = min(1.55, max(.65, (h-center[1])/max(1,width)))

        def paint(points, color, opacity=1.0, blur=1.1):
            self._blend_shape(image, poly(points), color, opacity, blur)

        def shade(f=1., side=0.):
            return self._adjust(base, scale*(f + direction*side))

        # Jacket silhouette. Body layer first; garment is intentionally stylized.
        paint([(-.53,-.045),(-.47,-.12),(-.31,-.12),(.31,-.12),(.47,-.12),
               (.53,-.045),(.68,.31),(.67,extent),(-.67,extent),(-.68,.31)], shade(.77))
        # White shirt under suit.
        paint([(-.27,-.115),(0,.045),(.27,-.115),(.19,extent),(-.19,extent)],
              self._adjust((222,231,237), min(1.10,scale)), .99)
        # Shirt center and subtle placket.
        paint([(-.012,.11),(.012,.11),(.025,extent),(-.015,extent)], (178,191,201), .4)
        # Jacket body panels / subtle illumination gradient.
        paint([(-.58,-.08),(-.27,-.12),(-.03,.44),(-.055,extent),(-.67,extent),(-.67,.31)],
              shade(1.02,-1), .995)
        paint([(.58,-.08),(.27,-.12),(.03,.44),(.055,extent),(.67,extent),(.67,.31)],
              shade(.97,1), .995)
        # Overlap dark seams / lapels give depth.
        if opts.shadows:
            paint([(-.27,-.12),(-.055,.44),(-.30,.27),(-.37,-.01)], shade(.54,-1), .55,2)
            paint([(.27,-.12),(.055,.44),(.30,.27),(.37,-.01)], shade(.54,1), .55,2)
        paint([(-.27,-.115),(-.075,.40),(-.31,.255),(-.39,-.02)], shade(1.29,-1), .98)
        paint([(.27,-.115),(.075,.40),(.31,.255),(.39,-.02)], shade(1.18,1), .98)
        # Tie, deliberately visible only around neck and upper chest.
        paint([(-.05,.04),(0,.11),(.05,.04),(.028,.18),(-.028,.18)], (36,31,89), 1.,.5)
        paint([(-.028,.18),(.028,.18),(.052,.50),(0,.59),(-.052,.50)], (35,31,88), 1.,.5)
        # Pocket line and jacket button (scaled with shoulder width).
        color = self._adjust((171,180,190), scale)
        pocket = poly([(.31,.52),(.50,.52)])
        cv2.line(image, tuple(pocket[0]), tuple(pocket[1]), color, max(1,int(width*.010)), cv2.LINE_AA)
        button_pt = poly([(.035,.77),(.035,.78),(.045,.78)])
        cv2.circle(image, tuple(button_pt[0]), max(2,int(width*.014)), (135,139,146), -1, cv2.LINE_AA)

        # Composite real arms in front if pose predicts an arm crossing chest.
        # Limitation: this is forearm-shaped occlusion, NOT a skin/hand mask.
        if opts.restore_forearms:
            m = self._segment_restore_mask((h,w),pose,measured_width).astype(np.float32)[:,:,None]/255
            current = image.astype(np.float32)
            image[:] = (current + (frame.astype(np.float32)-current)*m).astype(np.uint8)
        return image
