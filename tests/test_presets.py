import numpy as np
import pytest

from cognisign.features import centre_and_scale
from cognisign.presets import PRESET_GESTURES, classify_preset, finger_states
from tests.handgen import POSES, make_hand, pose


@pytest.mark.parametrize("name", sorted(POSES))
def test_each_preset_is_recognised(name):
    assert classify_preset(pose(name)) == name


@pytest.mark.parametrize("name", sorted(set(POSES) - {"thumbs_up", "thumbs_down"}))
@pytest.mark.parametrize("tilt", [-35, -15, 20, 40])
def test_tilt_does_not_change_shape_gestures(name, tilt):
    assert classify_preset(pose(name, rotate_deg=tilt)) == name


@pytest.mark.parametrize("name", sorted(POSES))
def test_left_hand(name):
    assert classify_preset(pose(name, mirror=True), handedness="Left") == name


@pytest.mark.parametrize("name", sorted(POSES))
def test_survives_landmark_jitter(name):
    rng = np.random.default_rng(42)
    hits = sum(classify_preset(pose(name, noise=0.02, rng=rng)) == name for _ in range(30))
    assert hits >= 27


def test_thumb_sideways_is_not_a_thumbs_gesture():
    sideways = make_hand(index=False, middle=False, ring=False, pinky=False, rotate_deg=45)
    assert classify_preset(sideways) is None


def test_finger_states_open_and_closed():
    assert finger_states(centre_and_scale(pose("open_palm"))).as_tuple() == (True,) * 5
    assert finger_states(centre_and_scale(pose("fist"))).as_tuple() == (False,) * 5


def test_every_documented_preset_is_tested():
    assert set(PRESET_GESTURES) == set(POSES)
