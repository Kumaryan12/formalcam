import numpy as np
from formalcam.app import synthetic_pose
from formalcam.render import GarmentRenderer, RenderOptions


def test_render_changes_torso_but_not_background_or_face():
    image=np.full((540,960,3),125,np.uint8)
    out=GarmentRenderer().render(image,synthetic_pose(960,540),RenderOptions())
    assert out.shape==image.shape
    assert out.dtype==np.uint8
    assert np.any(out[390,480]!=image[390,480]), 'torso should change'
    assert np.array_equal(out[100,50],image[100,50]), 'background should not change'
    assert np.array_equal(out[110,480],image[110,480]), 'face should not change'


def test_none_pose_returns_original():
    im=np.full((540,960,3),120,np.uint8)
    out=GarmentRenderer().render(im,None)
    assert np.array_equal(out,im)
    assert out is not im


def test_outfits_differ_and_original_is_untouched():
    image=np.full((540,960,3),170,np.uint8)
    pose=synthetic_pose(960,540)
    renderer=GarmentRenderer()
    navy=renderer.render(image,pose,RenderOptions(outfit='navy'))
    black=renderer.render(image,pose,RenderOptions(outfit='black'))
    assert np.any(navy!=black)
    assert np.all(image==170)


def test_dimmer_room_changes_shading():
    pose=synthetic_pose(960,540)
    dim=np.full((540,960,3),45,np.uint8)
    bright=np.full((540,960,3),220,np.uint8)
    a=GarmentRenderer().render(dim,pose,RenderOptions(lighting=True))
    b=GarmentRenderer().render(bright,pose,RenderOptions(lighting=True))
    assert b[380,380].mean() > a[380,380].mean()


def test_one_pixel_lighting_region_has_finite_direction():
    image = np.full((80, 1, 3), 120, np.uint8)
    scale, direction = GarmentRenderer()._estimate_light(image, 0., 60., 40.)
    assert np.isfinite(scale)
    assert np.isfinite(direction)


def test_brighter_screen_left_lights_left_jacket_panel():
    image = np.full((540, 960, 3), 60, np.uint8)
    image[:, :480] = 220
    pose = synthetic_pose(960, 540)
    lit = GarmentRenderer().render(image, pose, RenderOptions(lighting=True, shadows=False))
    flat = GarmentRenderer().render(image, pose, RenderOptions(lighting=False, shadows=False))
    left_gain = lit[400, 370].mean() / flat[400, 370].mean()
    right_gain = lit[400, 590].mean() / flat[400, 590].mean()
    assert left_gain > right_gain


def test_fit_can_lift_collar_and_narrow_garment_without_mutating_source():
    image = np.full((720, 1280, 3), 125, np.uint8)
    pose = synthetic_pose(1280, 720)
    default = GarmentRenderer().render(image, pose, RenderOptions(lighting=False))
    fitted = GarmentRenderer().render(image, pose, RenderOptions(lighting=False, width_scale=.85, vertical_offset=-.05))
    default_mask = np.any(default != image, axis=2)
    fitted_mask = np.any(fitted != image, axis=2)
    assert np.where(fitted_mask)[0].min() < np.where(default_mask)[0].min()
    assert fitted_mask[550].sum() < default_mask[550].sum()
    assert np.all(image == 125)


def test_restoring_forearm_outside_garment_preserves_original_pixels():
    from formalcam.pose import BodyPose, Point
    image = np.full((540, 960, 3), 125, np.uint8)
    pose = BodyPose(Point(336, 205), Point(624, 205), Point(30, 100), None, Point(100, 110))
    out = GarmentRenderer().render(image, pose, RenderOptions(restore_forearms=True))
    assert np.array_equal(out[:140], image[:140])


def test_lighting_reacquisition_uses_current_room_after_pose_loss():
    renderer = GarmentRenderer()
    pose = synthetic_pose(960, 540)
    dim = np.full((540, 960, 3), 45, np.uint8)
    bright = np.full((540, 960, 3), 220, np.uint8)
    renderer.render(dim, pose)
    renderer.render(bright, None)
    reacquired = renderer.render(bright, pose)
    fresh = GarmentRenderer().render(bright, pose)
    assert np.array_equal(reacquired, fresh)


def test_light_smoothing_response_is_consistent_at_15_and_30_fps():
    dim = np.full((540, 960, 3), 45, np.uint8)
    bright = np.full((540, 960, 3), 220, np.uint8)
    fast, slow = GarmentRenderer(), GarmentRenderer()
    fast._estimate_light(dim, 480, 205, 288)
    slow._estimate_light(dim, 480, 205, 288)
    for _ in range(6): fast_result = fast._estimate_light(bright, 480, 205, 288, 1/30)
    for _ in range(3): slow_result = slow._estimate_light(bright, 480, 205, 288, 1/15)
    assert np.allclose(fast_result, slow_result)
