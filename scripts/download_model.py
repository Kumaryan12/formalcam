"""Explicit, one-time download of Google's public MediaPipe pose model."""
from pathlib import Path
import tempfile
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
TARGET = Path(__file__).resolve().parent.parent / 'models' / 'pose_landmarker_lite.task'


def valid_model(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            expected = {'pose_detector.tflite', 'pose_landmarks_detector.tflite'}
            return expected <= set(archive.namelist()) and archive.testzip() is None
    except (OSError, BadZipFile):
        return False


def download_model(target: Path = TARGET) -> None:
    if valid_model(target):
        print('Already present:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        print('Downloading MediaPipe pose model from:', URL, flush=True)
        urlretrieve(URL, temporary)
        if not valid_model(temporary):
            raise RuntimeError('Downloaded model is incomplete or invalid; retry the download.')
        temporary.replace(target)
        print('Saved:', target, 'size:', target.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    download_model()
