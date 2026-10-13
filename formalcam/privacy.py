"""Fail-closed macOS process network restriction before loading pose libraries."""
from __future__ import annotations

import ctypes
from importlib.metadata import version
import sys

SAFE_MEDIAPIPE_VERSION = '0.10.21'
PROFILE = b'(version 1) (allow default) (deny network*)'
_restricted = False


def restrict_network() -> None:
    global _restricted
    if _restricted:
        return
    if sys.platform != 'darwin':
        raise RuntimeError('This local-only preview requires the macOS network sandbox.')
    try:
        sandbox = ctypes.CDLL('/usr/lib/libsandbox.dylib')
        sandbox.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                                        ctypes.POINTER(ctypes.c_char_p)]
        sandbox.sandbox_init.restype = ctypes.c_int
        sandbox.sandbox_free_error.argtypes = [ctypes.c_char_p]
        sandbox.sandbox_free_error.restype = None
        error = ctypes.c_char_p()
        result = sandbox.sandbox_init(PROFILE, 0, ctypes.byref(error))
        if result != 0:
            detail = error.value.decode(errors='replace') if error.value else 'unknown error'
            if error.value:
                sandbox.sandbox_free_error(error)
            raise RuntimeError(f'Could not deny runtime networking: {detail}')
    except OSError as exc:
        raise RuntimeError('macOS network sandbox unavailable; preview will not start.') from exc
    _restricted = True
    print('Runtime network access denied for this process.', flush=True)


def prepare_pose_runtime() -> None:
    installed = version('mediapipe')
    if installed != SAFE_MEDIAPIPE_VERSION:
        raise RuntimeError(f'MediaPipe {installed} is not approved for this no-telemetry prototype. '
                           'Reinstall this project\'s requirements (MediaPipe 0.10.21).')
    restrict_network()
