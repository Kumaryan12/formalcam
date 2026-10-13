import numpy as np

from formalcam.geometry import TorsoFitter
from formalcam.pose import BodyPose, Point


def test_hip_lean_deforms_hem_without_moving_neck_anchor():
    upright = BodyPose(Point(200, 200), Point(400, 200), left_hip=Point(220, 400), right_hip=Point(380, 400))
    leaning = BodyPose(Point(200, 200), Point(400, 200), left_hip=Point(280, 400), right_hip=Point(440, 400))
    a = TorsoFitter().fit(upright, (720, 1280))
    b = TorsoFitter().fit(leaning, (720, 1280))
    assert np.array_equal(a.points([(0., 0.)]), b.points([(0., 0.)]))
    assert b.points([(0., 1.05)])[0, 0] > a.points([(0., 1.05)])[0, 0]


def test_rotation_and_distance_change_geometry():
    a = TorsoFitter().fit(BodyPose(Point(200, 200), Point(400, 200)), (720, 1280))
    b = TorsoFitter().fit(BodyPose(Point(100, 100), Point(500, 300)), (720, 1280))
    points = b.points([(-.5, 0.), (.5, 0.)])
    assert points[1, 1] > points[0, 1]
    assert b.shoulder_width > a.shoulder_width


def test_missing_hips_transition_smoothly_and_tracking_loss_resets():
    pose = BodyPose(Point(200, 200), Point(400, 200), left_hip=Point(280, 400), right_hip=Point(440, 400))
    fitter = TorsoFitter()
    first = fitter.fit(pose, (720, 1280))
    missing = fitter.fit(BodyPose(pose.left_shoulder, pose.right_shoulder), (720, 1280))
    assert 0 < missing.lean < first.lean
    assert fitter.fit(None, (720, 1280)) is None
    reacquired = fitter.fit(BodyPose(pose.left_shoulder, pose.right_shoulder), (720, 1280))
    assert reacquired.lean == 0.


def test_out_of_frame_hips_use_shoulder_fallback():
    pose = BodyPose(Point(200, 200), Point(400, 200), left_hip=Point(200, 900), right_hip=Point(400, 900))
    fitted = TorsoFitter().fit(pose, (720, 1280))
    assert fitted.length_ratio == np.float32(1.05)
