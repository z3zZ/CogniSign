import json

import numpy as np
import pytest

from cognisign.classifier import GestureLibrary, KNNClassifier, LibraryError, Recogniser
from cognisign.features import FEATURE_SIZE, FEATURE_VERSION, extract_features
from tests.handgen import make_hand, pose


def _record(lib, label, n=20, seed=0, **pose_kw):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        lib.add_landmarks(label, make_hand(noise=0.02, rng=rng,
                                           rotate_deg=rng.uniform(-15, 15), **pose_kw))


@pytest.fixture
def library():
    lib = GestureLibrary()
    # two custom gestures the presets don't know about
    _record(lib, "water", thumb=False, index=True, middle=False, ring=False, pinky=True, seed=1)
    _record(lib, "eat", thumb=True, index=True, middle=True, ring=False, pinky=False, seed=2)
    return lib


def test_knn_learns_custom_gestures(library):
    knn = KNNClassifier(library)
    rng = np.random.default_rng(99)
    water = make_hand(thumb=False, middle=False, ring=False, noise=0.02, rng=rng)
    eat = make_hand(ring=False, pinky=False, noise=0.02, rng=rng)
    assert knn.predict(extract_features(water)).label == "water"
    assert knn.predict(extract_features(eat)).label == "eat"


def test_knn_rejects_unknown_pose(library):
    knn = KNNClassifier(library)
    pred = knn.predict(extract_features(pose("open_palm")))
    assert pred.label is None and pred.source == "none"


def test_empty_library_predicts_nothing():
    assert KNNClassifier(GestureLibrary()).predict(np.zeros(FEATURE_SIZE)).label is None


def test_recogniser_prefers_custom_then_falls_back_to_presets(library):
    rec = Recogniser(library)
    custom = rec.predict(make_hand(thumb=False, middle=False, ring=False))
    assert (custom.label, custom.source) == ("water", "custom")
    preset = rec.predict(pose("thumbs_up"))
    assert (preset.label, preset.source) == ("thumbs_up", "preset")


def test_custom_recording_can_override_a_preset():
    lib = GestureLibrary()
    _record(lib, "my_hello", seed=3)  # same shape as open_palm
    assert Recogniser(lib).predict(pose("open_palm")).label == "my_hello"


def test_presets_can_be_disabled():
    assert Recogniser(GestureLibrary(), use_presets=False).predict(pose("fist")).label is None


def test_save_load_roundtrip(tmp_path, library):
    path = tmp_path / "nested" / "gestures.json"
    library.save(path)
    loaded = GestureLibrary.load(path)
    assert loaded.counts() == library.counts()
    assert not list(path.parent.glob(".gestures-*"))  # temp file cleaned up


def test_load_missing_file_is_empty(tmp_path):
    assert GestureLibrary.load(tmp_path / "nope.json").labels == []


def test_old_feature_version_is_rejected(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"format": "cognisign-gestures",
                                "feature_version": FEATURE_VERSION - 1, "samples": {}}))
    with pytest.raises(LibraryError, match="re-record"):
        GestureLibrary.load(path)


def test_validation():
    lib = GestureLibrary()
    with pytest.raises(LibraryError):
        lib.add("  ", np.zeros(FEATURE_SIZE))
    with pytest.raises(LibraryError):
        lib.add("x", np.zeros(10))
    lib.add("x", np.zeros(FEATURE_SIZE))
    assert lib.remove("x") and not lib.remove("x")
