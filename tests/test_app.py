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
