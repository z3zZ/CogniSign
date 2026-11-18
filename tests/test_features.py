import numpy as np
import pytest

from cognisign.features import (
    FEATURE_SIZE,
    InvalidLandmarks,
    canonicalise,
    extract_features,
    to_array,
)
from tests.handgen import make_hand, pose


def test_feature_size():
    assert extract_features(make_hand()).shape == (FEATURE_SIZE,)


def test_position_and_scale_invariant():
    a = extract_features(make_hand(scale=0.08, offset=(0.2, 0.3)))
    b = extract_features(make_hand(scale=0.20, offset=(0.7, 0.8)))
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("deg", [-60, -20, 15, 45, 170])
def test_shape_is_tilt_invariant_but_orientation_is_kept(deg):
    shape0, dir0 = canonicalise(make_hand())
    shape1, dir1 = canonicalise(make_hand(rotate_deg=deg))
    np.testing.assert_allclose(shape0, shape1, atol=1e-9)
    angle = np.degrees(np.arccos(np.clip(dir0 @ dir1, -1, 1)))
    assert angle == pytest.approx(abs(deg) if abs(deg) <= 180 else 360 - abs(deg), abs=1e-6)


def test_left_hand_matches_right_hand_template():
    right = extract_features(pose("peace"))
    left = extract_features(pose("peace", mirror=True), handedness="Left")
    np.testing.assert_allclose(right, left, atol=1e-9)


def test_thumbs_up_and_down_are_far_apart():
    up = extract_features(pose("thumbs_up"))
    down = extract_features(pose("thumbs_down"))
    assert np.linalg.norm(up - down) > 2.0


def test_accepts_flat_and_2d_input():
    pts = make_hand()
    np.testing.assert_allclose(to_array(pts.ravel()), pts)
    assert to_array(pts[:, :2]).shape == (21, 3)


def test_accepts_mediapipe_style_objects():
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    pts = make_hand()
    objs = [P(*p) for p in pts]
    np.testing.assert_allclose(extract_features(objs), extract_features(pts))


@pytest.mark.parametrize("bad", [np.zeros((20, 3)), np.zeros((21, 4)), np.full((21, 3), np.nan)])
def test_rejects_bad_shapes(bad):
    with pytest.raises(InvalidLandmarks):
        extract_features(bad)


def test_rejects_degenerate_hand():
    with pytest.raises(InvalidLandmarks):
        extract_features(np.zeros((21, 3)))
