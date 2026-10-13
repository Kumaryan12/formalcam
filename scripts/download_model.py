"""Explicit, one-time download of Google's public MediaPipe pose model."""
from pathlib import Path
import argparse
import tempfile
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
TARGET = Path(__file__).resolve().parent.parent / 'models' / 'pose_landmarker_lite.task'
SEGMENTATION_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'
SEGMENTATION_TARGET = TARGET.parent / 'selfie_multiclass_256x256.tflite'


def valid_model(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            expected = {'pose_detector.tflite', 'pose_landmarks_detector.tflite'}
            return expected <= set(archive.namelist()) and archive.testzip() is None
    except (OSError, BadZipFile):
        return False


def valid_tflite(path: Path) -> bool:
    try:
        with path.open('rb') as stream:
            stream.seek(4)
            return path.stat().st_size > 1024 and stream.read(4) == b'TFL3'
    except OSError:
        return False


def download_model(target: Path = TARGET, url: str = URL) -> None:
    validator = valid_tflite if target.suffix == '.tflite' else valid_model
    if validator(target):
        print('Already present:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        print('Downloading MediaPipe model from:', url, flush=True)
        urlretrieve(url, temporary)
        if not validator(temporary):
            raise RuntimeError('Downloaded model is incomplete or invalid; retry the download.')
        temporary.replace(target)
        print('Saved:', target, 'size:', target.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explicit local MediaPipe model download')
    parser.add_argument('--segmentation', action='store_true', help='Download the optional Apache-2.0 SelfieMulticlass model')
    args = parser.parse_args()
    download_model(SEGMENTATION_TARGET, SEGMENTATION_URL) if args.segmentation else download_model()
