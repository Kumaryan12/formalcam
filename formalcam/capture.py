"""Optional single-reader camera pump with one latest frame and no video queue."""
from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np


class Camera(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class LatestFrameCapture:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest: tuple[np.ndarray, float] | None = None
        self.received = 0
        self.delivered = 0
        self.skipped = 0
        self.last_sequence = 0
        self.error: str | None = None
        self.thread = threading.Thread(target=self._pump, name='formalcam-camera', daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        try:
            while not self.stopped.is_set():
                ok, image = self.camera.read()
                if not ok or image is None:
                    raise RuntimeError('Camera stopped supplying frames.')
                timestamp = time.monotonic()
                with self.condition:
                    self.latest = image, timestamp
                    self.received += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            self.camera.release()

    def read(self, timeout: float = 2.) -> tuple[np.ndarray, float]:
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.received > self.last_sequence or self.error is not None,
                timeout=timeout)
            if not ready:
                raise RuntimeError('Camera did not deliver a new frame within two seconds.')
            if self.received == self.last_sequence:
                raise RuntimeError(self.error or 'Camera reader stopped.')
            assert self.latest is not None
            self.skipped += max(0, self.received - self.last_sequence - 1)
            self.last_sequence = self.received
            self.delivered += 1
            return self.latest

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError('Camera reader did not stop; close and restart FormalCam.')
