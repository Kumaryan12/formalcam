from argparse import Namespace
from io import StringIO
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from formalcam import app
from formalcam.diagnostics import FrameStats


def test_fps_measures_completed_loop_time_including_display():
    output = StringIO()
    stats = FrameStats(10., output)
    times = dict(capture=.01, pose=.01, render=.01, send=0., display=.02)
    stats.update(10.05, True, times)
    stats.update(10.10, False, times)
    assert stats.fps == pytest.approx(20.)
    assert stats.pose_frames == 1
    assert 'display_ms' in output.getvalue()


@pytest.mark.parametrize('failure', ['closed', 'read', 'virtual'])
def test_camera_and_landmarker_are_released_on_startup_failure(monkeypatch, failure):
    released = []

    class Landmarker:
        def __enter__(self): return self
        def __exit__(self, *args): released.append('pose')

    class Capture:
        def isOpened(self): return failure != 'closed'
        def set(self, *args): pass
        def get(self, *args): return 30.
        def read(self): return (False, None) if failure == 'read' else (True, np.zeros((20, 30, 3), np.uint8))
        def release(self): released.append('camera')

    def fail_sender(**kwargs):
        raise RuntimeError('backend missing')

    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'pyvirtualcam', SimpleNamespace(Camera=fail_sender, PixelFormat=SimpleNamespace(RGB=1)))
    monkeypatch.setattr(app, 'create_landmarker', lambda _: Landmarker())
    monkeypatch.setattr(app.cv2, 'VideoCapture', lambda *args: Capture())
    monkeypatch.setattr(app.cv2, 'destroyAllWindows', lambda: released.append('windows'))
    args = Namespace(camera=0, width=1280, height=720, virtual_camera=failure == 'virtual')
    with pytest.raises(RuntimeError):
        app.run(args)
    assert released == ['camera', 'windows', 'pose']


def test_missing_pose_sends_clean_original_and_video_timestamps_increase(monkeypatch, tmp_path):
    timestamps, sent, closed = [], [], []
    raw = np.full((20, 30, 3), 7, np.uint8)

    class Landmarker:
        def __enter__(self): return self
        def __exit__(self, *args): closed.append('pose')
        def detect_for_video(self, image, timestamp):
            timestamps.append(timestamp)
            return SimpleNamespace(pose_landmarks=[])

    class Capture:
        def isOpened(self): return True
        def set(self, *args): pass
        def get(self, *args): return 30.
        def read(self): return True, raw.copy()
        def release(self): closed.append('camera')

    class Sender:
        device = 'test'
        def send(self, image): sent.append(image.copy())
        def close(self): closed.append('sender')

    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace(Image=lambda **kwargs: kwargs['data'], ImageFormat=SimpleNamespace(SRGB=1)))
    monkeypatch.setitem(sys.modules, 'pyvirtualcam', SimpleNamespace(Camera=lambda **kwargs: Sender(), PixelFormat=SimpleNamespace(RGB=1)))
    monkeypatch.setattr(app, 'create_landmarker', lambda _: Landmarker())
    monkeypatch.setattr(app.cv2, 'VideoCapture', lambda *args: Capture())
    monkeypatch.setattr(app.cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(app.cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(app.cv2, 'waitKey', lambda *args: -1)
    monkeypatch.setattr(app.cv2, 'getWindowProperty', lambda *args: 1.)
    monkeypatch.setattr(app, 'badge', lambda image, lines: image.fill(99))
    monkeypatch.setattr(app.time, 'monotonic', lambda: 100.)
    args = Namespace(camera=0, width=30, height=20, virtual_camera=True,
                     profile=None, no_mirror=False, fit_width=1., fit_offset=0., latest_frame=False,
                     max_frames=3, snapshot_frame=0, snapshot_dir=tmp_path)
    app.run(args)
    assert timestamps == [0, 1, 2]
    assert len(sent) == 3 and all(np.array_equal(frame, raw) for frame in sent)
    assert closed == ['sender', 'camera', 'pose']
