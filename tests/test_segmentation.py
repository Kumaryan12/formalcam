import threading

import numpy as np
import pytest

from formalcam.masks import clothing_allowance, composite_garment, propagate_mask, protect_landmarks
from formalcam.pose import BodyPose, Point
from formalcam.segmentation import SegmentationEngine

LABELS = ['background','hair','body-skin','face-skin','clothes','others']


def test_only_clothes_are_eligible_with_other_categories_preserved():
    classes = np.tile(np.arange(6), (8,1))
    maps = [(classes == i).astype(np.float32) for i in range(6)]
    mask = clothing_allowance(maps, LABELS)
    assert np.all(mask[:, :4] == 0)
    assert np.all(mask[:, 5] == 0)
    assert np.all(mask[:, 4] > 0)
    original = np.full((8,6,3), (17,33,91), np.uint8)
    rendered = np.full_like(original, (105,61,37))
    output = composite_garment(original, rendered, mask)
    assert np.array_equal(output[:, :4], original[:, :4])
    assert np.array_equal(output[:, 5], original[:, 5])
    assert output.dtype == np.uint8 and np.all(original == [17,33,91])


def test_wrong_model_labels_and_nonfinite_masks_are_rejected():
    maps = [np.ones((8,8),np.float32) for _ in range(6)]
    with pytest.raises(ValueError): clothing_allowance(maps, ['wrong']*6)
    maps[4][0,0] = np.nan
    with pytest.raises(ValueError): clothing_allowance(maps, LABELS)


def test_half_alpha_blend_preserves_bgr_channel_order_and_source():
    original = np.full((12,16,3), [10,30,90], np.uint8)
    garment = np.full_like(original, [110,70,30])
    output = composite_garment(original, garment, np.full((3,4), .5, np.float32))
    assert np.all(output == [60,50,60])
    assert np.all(original == [10,30,90])


def test_scene_change_drops_cached_mask_and_new_motion_restores_original_region():
    original = np.full((90,160),100,np.uint8)
    assert propagate_mask(np.ones_like(original,np.float32), original, np.full_like(original,255)) is None
    moved = original.copy()
    moved[30:50,60:80] = 0
    propagated = propagate_mask(np.ones_like(original,np.float32), original, moved)
    assert propagated is not None and propagated[40,70] == 0.


def test_head_and_current_wrist_have_extra_protection():
    pose = BodyPose(Point(200,250),Point(400,250), left_wrist=Point(300,400), nose=Point(300,170))
    protected = protect_landmarks(np.ones((720,1280),np.float32), pose, (720,1280))
    assert protected[170,300] == 0 and protected[400,300] == 0
    assert protected[600,300] == 1


def test_missing_model_is_optional_and_returns_no_unsafe_mask(tmp_path):
    engine = SegmentationEngine(tmp_path/'missing.tflite')
    assert engine.error and engine.update(np.zeros((90,160,3),np.uint8),0,1.,1.).allowance is None
    engine.close()


def test_only_one_job_is_in_flight_and_lost_pose_discards_result():
    started, release = threading.Event(), threading.Event()
    calls = []
    def predict(image, timestamp):
        calls.append(timestamp)
        started.set()
        release.wait(2.)
        return np.ones(image.shape[:2],np.float32)
    engine = SegmentationEngine(predictor=predict)
    frame = np.full((90,160,3),100,np.uint8)
    try:
        assert engine.update(frame,1,1.,1.).allowance is None
        assert started.wait(1.)
        engine.update(frame,2,1.02,1.02)
        assert calls == [1]
        engine.reset()
        release.set()
        engine._future.result(timeout=2.)
        assert engine.update(frame,3,1.04,1.04).allowance is None
    finally:
        release.set()
        engine.close()


def test_expired_masks_are_not_composited():
    engine = SegmentationEngine(predictor=lambda image, stamp: np.ones(image.shape[:2],np.float32))
    frame = np.full((90,160,3),100,np.uint8)
    try:
        engine.update(frame,1,1.,1.)
        engine._future.result(timeout=2.)
        accepted = engine.update(frame,2,1.01,1.01)
        assert accepted.allowance is not None and accepted.fresh
        expired = engine.update(frame,3,1.5,1.5)
        assert expired.allowance is None
    finally:
        engine.close()


def test_fresh_protected_region_overrides_temporal_history():
    next_mask = [1.]
    engine = SegmentationEngine(predictor=lambda image, stamp: np.full(image.shape[:2],next_mask[0],np.float32))
    frame = np.full((90,160,3),100,np.uint8)
    try:
        engine.update(frame,1,1.,1.)
        engine._future.result(timeout=2.)
        assert engine.update(frame,2,1.01,1.01).allowance.mean() == 1.
        next_mask[0] = 0.
        engine.update(frame,3,1.12,1.12)
        engine._future.result(timeout=2.)
        assert engine.update(frame,4,1.13,1.13).allowance.max() == 0.
    finally:
        engine.close()
