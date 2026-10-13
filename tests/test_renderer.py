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
