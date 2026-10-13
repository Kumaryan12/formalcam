import threading

import numpy as np
import pytest

from formalcam.capture import LatestFrameCapture


def test_reader_delivers_latest_frame_instead_of_queued_old_frames():
    ended = threading.Event()

    class Camera:
        sequence = 0
        def read(self):
            self.sequence += 1
            if self.sequence > 3: return False, None
            return True, np.full((2, 2, 3), self.sequence, np.uint8)
        def release(self): ended.set()

    reader = LatestFrameCapture(Camera())
    assert ended.wait(2.)
    image, timestamp = reader.read()
    assert np.all(image == 3)
    assert reader.skipped == 2 and reader.delivered == 1
    assert timestamp > 0.
    with pytest.raises(RuntimeError, match='stopped supplying'):
        reader.read()
    reader.close()


def test_reader_reports_camera_exception_and_releases_it():
    released = threading.Event()

    class Camera:
        def read(self): raise OSError('disconnected')
        def release(self): released.set()

    reader = LatestFrameCapture(Camera())
    with pytest.raises(RuntimeError, match='disconnected'):
        reader.read()
    reader.close()
    assert released.is_set()
