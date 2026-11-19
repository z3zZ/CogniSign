"""End-to-end: fake camera + fake tracker through the real engine, HUD and loop."""

import numpy as np
import pytest

from cognisign.actions import Action, ActionDispatcher
from cognisign.app import GestureEngine, run
from cognisign.classifier import GestureLibrary
from cognisign.cli import main
from cognisign.config import load_config, parse_config
from cognisign.hud import HudState, draw
from cognisign.recorder import SampleCollector
from cognisign.tracker import Hand, to_hands
from tests.handgen import pose
from tests.test_actions import FakeKeyboard, FakeSpeaker


def hand(name, **kw):
    lms = pose(name, **kw)
    return Hand(landmarks=lms, pixels=(lms[:, :2] * [640, 480]).astype(int),
                handedness="Right", score=0.95)


class ScriptedTracker:
    """Plays back a list of poses, one per frame."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def process(self, frame, ts):
        name = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [hand(name)] if name else []


class FakeClock:
    def __init__(self, step=1 / 30):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture
def engine():
    cfg = load_config("/nonexistent/config.yaml")
    d = ActionDispatcher(FakeSpeaker(), FakeKeyboard())
    return GestureEngine(cfg, d, library=GestureLibrary())


def test_holding_thumbs_up_says_yes_once(engine):
    script = [None] * 5 + ["thumbs_up"] * 60 + [None] * 5
    frames = (np.zeros((480, 640, 3), np.uint8) for _ in script)
    run(engine.config, engine, ScriptedTracker(script), frames, show=False, clock=FakeClock())
    assert engine.dispatcher.speaker.said == ["Yes"]


def test_build_and_speak_a_sentence(engine):
    cfg = parse_config({"recognition": {"hold_ms": 300, "cooldown_ms": 0}, "gestures": {
        "point": {"action": "phrase_add", "value": "I want"},
        "peace": {"action": "phrase_add", "value": "water"},
        "fist": {"action": "phrase_speak"},
    }})
    eng = GestureEngine(cfg, ActionDispatcher(FakeSpeaker(), FakeKeyboard()), GestureLibrary())
    script = ["point"] * 15 + [None] * 5 + ["peace"] * 15 + [None] * 5 + ["fist"] * 15
    frames = (np.zeros((480, 640, 3), np.uint8) for _ in script)
    run(cfg, eng, ScriptedTracker(script), frames, show=False, clock=FakeClock())
    assert eng.dispatcher.speaker.said == ["I want water."]


def test_pause_gesture_suppresses_others(engine):
    run_script(engine, ["rock"] * 30 + [None] * 3 + ["thumbs_up"] * 30)
    assert engine.dispatcher.speaker.said == []
    assert engine.hud.paused


def test_hud_draws_every_element_without_error():
    for w, h in [(640, 480), (1280, 720), (320, 240)]:
        frame = np.zeros((h, w, 3), np.uint8)
        state = HudState(label="thumbs_up", source="preset", confidence=1.0, progress=0.6,
                         mode="sentence", phrase_words=["can", "I", "have", "water"] * 8,
                         toast="Added: water", toast_kind="added", toast_alpha=0.7, flash=1.0,
                         bindings=[("thumbs_up", 'add "Yes"'), ("fist", "speak sentence")] * 6,
                         fired_label="thumbs_up")
        out = draw(frame, [hand("open_palm", offset=(0.5, 0.7), scale=0.15)], state)
        assert out.shape == (h, w, 3) and out.any()
        draw(frame, [], HudState(paused=True, empty_hint="Make the ok sign", show_bindings=False))


def run_script(eng, script):
    """Feed one pose per ~30fps frame, continuing from the engine's current clock."""
    t = eng._now + 33
    for name in script:
        eng.step([hand(name)] if name else [], t)
        t += 33


def test_default_flow_builds_a_sentence_with_mode_switch(engine):
    # ok -> sentence mode, thumbs up adds "Yes", peace adds "Thank you", fist speaks
    run_script(engine, ["ok"] * 30 + [None] * 5 + ["thumbs_up"] * 30 + [None] * 5
               + ["peace"] * 30 + [None] * 5)
    assert engine.hud.mode == "sentence"
    assert engine.hud.phrase_words == ["Yes", "Thank you"]
    assert engine.dispatcher.speaker.said == []
    run_script(engine, ["fist"] * 30)
    assert engine.dispatcher.speaker.said == ["Yes Thank you."]


def test_every_action_shows_a_toast_and_flash(engine):
    run_script(engine, ["thumbs_up"] * 25)
    assert engine.hud.toast == "Said: Yes" and engine.hud.flash == 1.0
    assert engine.hud.fired_label == "thumbs_up"
    run_script(engine, [None] * 5 + ["fist"] * 25)
    assert engine.hud.toast == "Sentence is empty, nothing to say"
    assert engine.hud.toast_kind == "warn"


def test_toast_fades_out(engine):
    run_script(engine, ["thumbs_up"] * 25)
    assert engine.hud.toast_alpha == 1.0
    run_script(engine, [None] * 80)  # ~2.6 s later
    assert engine.hud.toast_alpha == 0.0 and engine.hud.flash == 0.0


def test_unbound_custom_gesture_warns_instead_of_silence():
    lib = GestureLibrary()
    for _ in range(10):
        lib.add_landmarks("wave", pose("open_palm"))
    cfg = load_config("/nonexistent/config.yaml")
    eng = GestureEngine(cfg, ActionDispatcher(FakeSpeaker(), FakeKeyboard()), lib)
    run_script(eng, ["open_palm"] * 25)
    assert "isn't bound" in eng.hud.toast
    assert ("wave", "(not set)") in eng.hud.bindings


def test_bindings_reflect_current_mode(engine):
    engine.step([], 0)
    rows = dict(engine.hud.bindings)
    assert rows["thumbs_up"] == 'say "Yes"' and rows["ok"] == "quick <-> sentence"
    assert list(rows)[0] == "open_palm"  # presets listed in a stable order
    engine.manual(Action("toggle_mode"))
    assert dict(engine.hud.bindings)["thumbs_up"] == 'add "Yes"'
    assert engine.hud.toast.startswith("Sentence mode")


def test_empty_hint_tells_you_how_to_start(engine):
    engine.step([], 0)
    assert engine.hud.empty_hint == "Make the ok sign to start building a sentence"
    engine.manual(Action("toggle_mode"))
    assert "fist to speak it" in engine.hud.empty_hint


def test_keyboard_shortcuts_give_feedback(engine):
    engine.step([], 0)
    engine.manual(Action("toggle_pause"))
    assert engine.hud.paused and engine.hud.toast == "Paused"
    engine.toggle_bindings()
    assert engine.hud.show_bindings is False


def test_sample_collector_spacing_and_countdown():
    c = SampleCollector("water", target=5, interval_ms=100, countdown_ms=1000)
    h = hand("peace")
    captured = [c.update(h, t) for t in range(0, 3000, 20)]
    assert sum(captured) == 5 and c.done
    assert c.phase(3000) == "done"
    lib = GestureLibrary()
    assert c.commit(lib) == 5 and lib.counts() == {"water": 5}


def test_to_hands_corrects_aspect_ratio():
    class LM:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Cat:
        category_name, score = "Left", 0.9

    class Result:
        hand_landmarks = [[LM(0.5, 0.5)] * 21]
        handedness = [[Cat()]]

    (h,) = to_hands(Result(), 1280, 720)
    assert h.handedness == "Left" and h.score == 0.9
    assert h.landmarks[0, 0] == pytest.approx(0.5 * 1280 / 720)
    assert tuple(h.pixels[0]) == (640, 360)


def test_cli_gestures_and_forget(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COGNISIGN_HOME", str(tmp_path))
    assert main(["init"]) == 0
    lib = GestureLibrary()
    lib.add_landmarks("water", pose("peace"))
    lib.save(tmp_path / "gestures.json")
    assert main(["gestures"]) == 0
    out = capsys.readouterr().out
    assert "thumbs_up" in out and "water" in out
    assert main(["forget", "water"]) == 0
    assert main(["forget", "water"]) == 1


def test_cli_reports_bad_config(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("camera: {zoom: 3}\n")
    assert main(["--config", str(p), "gestures"]) == 2
    assert "unknown setting" in capsys.readouterr().err
