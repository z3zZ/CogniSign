import pytest

from cognisign.stabiliser import GestureStabiliser


def feed(stab, frames, step=33, start=0):
    """frames: list of labels at ~30fps; returns list of (t, label) events."""
    events = []
    t = start
    for label in frames:
        ev = stab.update(label, t)
        if ev:
            events.append((t, ev.label))
        t += step
    return events


def test_fires_once_after_hold():
    stab = GestureStabiliser(hold_ms=600, cooldown_ms=0)
    events = feed(stab, ["yes"] * 60)  # 2 seconds held
    assert [e[1] for e in events] == ["yes"]
    assert 600 <= events[0][0] <= 700


def test_quick_pass_through_does_not_fire():
    stab = GestureStabiliser(hold_ms=600)
    assert feed(stab, [None] * 5 + ["yes"] * 8 + [None] * 10) == []


def test_must_release_before_firing_again():
    stab = GestureStabiliser(hold_ms=300, cooldown_ms=0, grace_ms=50)
    frames = ["yes"] * 20 + [None] * 5 + ["yes"] * 20
    assert [e[1] for e in feed(stab, frames)] == ["yes", "yes"]


def test_grace_period_forgives_a_dropped_frame():
    stab = GestureStabiliser(hold_ms=600, grace_ms=150)
    frames = ["yes"] * 10 + [None, None] + ["yes"] * 15
    assert [e[1] for e in feed(stab, frames)] == ["yes"]


def test_cooldown_between_different_gestures():
    stab = GestureStabiliser(hold_ms=100, cooldown_ms=1000, grace_ms=0)
    events = feed(stab, ["a"] * 6 + ["b"] * 10 + ["b"] * 30)
    assert [e[1] for e in events] == ["a", "b"]
    assert events[1][0] - events[0][0] >= 1000


def test_progress_rises_then_resets():
    stab = GestureStabiliser(hold_ms=600, grace_ms=0)
    stab.update("yes", 0)
    stab.update("yes", 300)
    assert stab.progress == pytest.approx(0.5)
    stab.update("yes", 700)  # fires
    assert stab.progress == 0.0 and not stab.armed
    stab.update(None, 800)
    assert stab.armed


def test_negative_timings_rejected():
    with pytest.raises(ValueError):
        GestureStabiliser(hold_ms=-1)
