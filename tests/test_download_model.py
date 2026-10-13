from zipfile import ZipFile

import pytest

from scripts import download_model as downloader


def test_interrupted_download_does_not_replace_existing_model(tmp_path, monkeypatch):
    target = tmp_path / 'model.task'
    target.write_bytes(b'old incomplete model')

    def interrupted(url, destination):
        destination.write_bytes(b'partial download')
        raise OSError('connection interrupted')

    monkeypatch.setattr(downloader, 'urlretrieve', interrupted)
    with pytest.raises(OSError):
        downloader.download_model(target)
    assert target.read_bytes() == b'old incomplete model'
    assert list(tmp_path.iterdir()) == [target]


def test_incomplete_large_download_is_retried(tmp_path, monkeypatch):
    target = tmp_path / 'model.task'
    target.write_bytes(b'x' * 200_000)

    def complete(url, destination):
        with ZipFile(destination, 'w') as archive:
            archive.writestr('pose_detector.tflite', b'detector')
            archive.writestr('pose_landmarks_detector.tflite', b'landmarks')

    monkeypatch.setattr(downloader, 'urlretrieve', complete)
    downloader.download_model(target)
    assert downloader.valid_model(target)

    def forbidden(*args): raise AssertionError('Valid models must not be redownloaded')
    monkeypatch.setattr(downloader, 'urlretrieve', forbidden)
    downloader.download_model(target)
