"""FormalCam local webcam preview and optional virtual-camera sender.

Controls: 1/2/3 palette | L lighting | S shading | A arms | O overlay | Q quit.
"""
from __future__ import annotations
import argparse
import contextlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .diagnostics import FrameStats, save_snapshot
from .pose import BodyPose, Point, PoseSmoother, from_mediapipe
from .render import GarmentRenderer, RenderOptions

MODEL = Path(__file__).resolve().parent.parent / 'models' / 'pose_landmarker_lite.task'


def synthetic_pose(w: int, h: int) -> BodyPose:
    return BodyPose(Point(w*.35,h*.38), Point(w*.65,h*.38),
                    Point(w*.28,h*.57), Point(w*.72,h*.57),
                    Point(w*.39,h*.76), Point(w*.61,h*.76))


def create_landmarker(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing pose model at {model_path}; run python scripts/download_model.py first.')
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=.52,
        min_tracking_confidence=.55,
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def badge(image: np.ndarray, lines: list[str]):
    for i,line in enumerate(lines):
        y=24 + 23*i
        cv2.putText(image,line,(16,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,line,(16,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(247,247,247),1,cv2.LINE_AA)


def demo(out: Path):
    h,w=540,960
    # Neutral placeholder, deliberately not a real person. This exercises rendering only.
    x=np.linspace(0,1,w,dtype=np.float32)
    base=np.empty((h,w,3),np.uint8)
    base[:]=[49,47,44]
    base[:]=np.clip(base.astype(np.float32)*(0.8+.30*x)[None,:,None],0,255).astype(np.uint8)
    cv2.ellipse(base,(w//2,150),(83,105),0,0,360,(135,172,197),-1,cv2.LINE_AA)
    cv2.ellipse(base,(w//2,485),(225,280),0,0,360,(135,124,108),-1,cv2.LINE_AA)
    out_img=GarmentRenderer().render(base,synthetic_pose(w,h),RenderOptions())
    badge(out_img,['FormalCam geometric demo (NOT photorealistic)','Real webcam preview: python -m formalcam.app'])
    out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out),out_img):
        raise OSError(f'Could not write demo image: {out}')
    print('Saved',out)


def run(args: argparse.Namespace) -> None:
    import mediapipe as mp

    # Initialize the local model before acquiring camera or virtual-camera resources.
    with contextlib.ExitStack() as stack:
        landmarker = stack.enter_context(create_landmarker(MODEL))
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
        stack.callback(cv2.destroyAllWindows)
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError('Could not open webcam. Close FaceTime/Zoom and allow camera access for the app hosting Python in macOS Settings > Privacy & Security > Camera.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError('Webcam opened, but no frames received.')
        h, w = first.shape[:2]
        print(f'Camera {args.camera}: {w}x{h}, reported {cap.get(cv2.CAP_PROP_FPS):.1f} fps', flush=True)
        sender = None
        if args.virtual_camera:
            try:
                import pyvirtualcam
                sender = pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.RGB)
                stack.callback(sender.close)
                print('Virtual camera started:', sender.device, flush=True)
            except Exception as exc:
                raise RuntimeError('Virtual camera unavailable. Use preview without --virtual-camera. OBS 30+ and pyvirtualcam 0.14+ are required on modern macOS; Start Virtual Camera once, then Stop it and quit OBS. Details: ' + str(exc)) from exc
        stream = None
        if args.profile:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(args.profile.open('w', newline=''))
        stats = FrameStats(time.monotonic(), stream)
        opts, renderer, smoother = RenderOptions(), GarmentRenderer(), PoseSmoother()
        overlay_on = True
        last_timestamp = -1
        window = 'FormalCam - Local Preview'
        print('Controls: 1/2/3 outfit, L lighting, S shadows, A arm restore, O original, P local snapshot, Q quit', flush=True)
        try:
            while True:
                begin = time.monotonic()
                ok, raw = cap.read()
                captured = time.monotonic()
                if not ok or raw is None:
                    raise RuntimeError('Camera stopped supplying frames.')
                if raw.shape[:2] != (h, w):
                    raise RuntimeError('Camera frame dimensions changed; restart the preview.')
                img = cv2.flip(raw, 1) if not args.no_mirror else raw
                rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(last_timestamp + 1, int((captured - stats.start) * 1000))
                last_timestamp = timestamp
                result = landmarker.detect_for_video(mp_image, timestamp)
                pose = from_mediapipe(result.pose_landmarks[0], w, h) if result.pose_landmarks else None
                pose = smoother.update(pose)
                posed = time.monotonic()
                processed = renderer.render(img, pose, opts) if overlay_on else img.copy()
                rendered = time.monotonic()
                if sender is not None:
                    sender.send(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                sent = time.monotonic()
                preview = processed.copy()
                badge(preview, [f'FormalCam 0.1 | {opts.outfit} | pose: {"yes" if pose else "no"} | {stats.fps:.1f} fps',
                                '1/2/3 outfit  L light  S shadow  A arm  O original  P save  Q quit',
                                '2D geometric demo - not a photorealistic garment'])
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                displayed = time.monotonic()
                stats.update(displayed, pose is not None,
                             dict(capture=captured-begin, pose=posed-captured,
                                  render=rendered-posed, send=sent-rendered,
                                  display=displayed-sent))
                if key in (ord('p'), ord('P')) or stats.frames == args.snapshot_frame:
                    save_snapshot(args.snapshot_dir, stats.frames, img, preview)
                if stats.frames % 60 == 0:
                    print(stats.summary(), flush=True)
                    if stream:
                        stream.flush()
                if key in (ord('q'), 27) or (args.max_frames and stats.frames >= args.max_frames):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('1'), ord('2'), ord('3')):
                    opts.outfit = {'1': 'navy', '2': 'charcoal', '3': 'black'}[chr(key)]
                if key in (ord('l'), ord('L')): opts.lighting = not opts.lighting
                if key in (ord('s'), ord('S')): opts.shadows = not opts.shadows
                if key in (ord('a'), ord('A')): opts.restore_forearms = not opts.restore_forearms
                if key in (ord('o'), ord('O')): overlay_on = not overlay_on
        finally:
            print('Session:', stats.summary(), flush=True)


def main():
    ap=argparse.ArgumentParser(description='FormalCam: experimental local webcam suit overlay')
    ap.add_argument('--camera',type=int,default=0,help='camera index')
    ap.add_argument('--width',type=int,default=1280)
    ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--no-mirror',action='store_true')
    ap.add_argument('--virtual-camera',action='store_true',help='Send via OBS virtual camera component')
    ap.add_argument('--demo-frame',action='store_true',help='Render synthetic preview without webcam/MediaPipe')
    ap.add_argument('--demo-output',type=Path,default=Path('demo_preview.png'))
    ap.add_argument('--profile', type=Path, help='Local numeric timing CSV; contains no images')
    ap.add_argument('--max-frames', type=int, default=0, help='Stop after N frames; 0 runs until Q')
    ap.add_argument('--snapshot-dir', type=Path, default=Path('diagnostics'), help='Local P-key snapshots')
    ap.add_argument('--snapshot-frame', type=int, default=0, help='Save a local still at frame N; 0 disables')
    args=ap.parse_args()
    if args.width <= 0 or args.height <= 0 or args.max_frames < 0 or args.snapshot_frame < 0:
        ap.error('Dimensions must be positive and frame counts nonnegative')
    try:
        if args.demo_frame: demo(args.demo_output)
        else: run(args)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ImportError, ValueError, cv2.error) as exc:
        print(f'FormalCam: {exc}', file=sys.stderr)
        raise SystemExit(1) from None

if __name__=='__main__':main()
