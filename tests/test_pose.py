from types import SimpleNamespace

import pytest

from formalcam.pose import BodyPose, Point, PoseSmoother, from_mediapipe


def landmarks():
    points = [SimpleNamespace(x=.5, y=.5, visibility=1., presence=1.) for _ in range(33)]
    points[11].x, points[12].x = .3, .7
    return points


def test_missing_pose_clears_smoothing_and_reacquires_without_stale_coordinates():
    smoother = PoseSmoother()
    smoother.update(BodyPose(Point(100, 100), Point(300, 100)))
    assert smoother.update(None) is None
    pose = BodyPose(Point(400, 200), Point(600, 200))
    assert smoother.update(pose) is pose


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_arm_landmarks_are_ignored(value):
    points = landmarks()
    points[13].x = value
    pose = from_mediapipe(points, 1280, 720)
    assert pose is not None
    assert pose.left_elbow is None


def test_partial_landmarks_fall_back_to_original():
    assert from_mediapipe([], 1280, 720) is None


def test_low_confidence_shoulder_rejects_pose():
    points = landmarks()
    points[11].presence = .2
    assert from_mediapipe(points, 1280, 720) is None
