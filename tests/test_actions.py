import pytest

from cognisign.actions import Action, ActionDispatcher
from cognisign.phrase import PhraseBuffer


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)

    def close(self):
        pass


class FakeKeyboard:
    def __init__(self):
        self.typed, self.pressed = [], []

    def type_text(self, text):
        self.typed.append(text)

    def hotkey(self, combo):
        self.pressed.append(combo)


@pytest.fixture
def d():
    return ActionDispatcher(FakeSpeaker(), FakeKeyboard(), PhraseBuffer())


def test_speak_type_hotkey(d):
    d.dispatch(Action("speak", "Yes"))
    d.dispatch(Action("type", "hello"))
    d.dispatch(Action("hotkey", "ctrl+c"))
    assert d.speaker.said == ["Yes"]
    assert d.keyboard.typed == ["hello"]
    assert d.keyboard.pressed == ["ctrl+c"]


def test_sentence_builder_flow(d):
    for w in ["can", "I", "have", "water"]:
        d.dispatch(Action("phrase_add", w))
    d.dispatch(Action("phrase_undo"))
    d.dispatch(Action("phrase_add", "some water please"))
    res = d.dispatch(Action("phrase_speak"))
    assert res.output == "Can I have some water please."
    assert d.speaker.said == ["Can I have some water please."]
    assert len(d.phrase) == 0


def test_speaking_empty_sentence_is_silent(d):
    d.dispatch(Action("phrase_speak"))
    assert d.speaker.said == []


def test_pause_blocks_everything_but_unpause(d):
    d.dispatch(Action("toggle_pause"))
    assert d.dispatch(Action("speak", "Yes")) is None
    d.dispatch(Action("toggle_pause"))
    d.dispatch(Action("speak", "Yes"))
    assert d.speaker.said == ["Yes"]


def test_none_action_and_missing_action(d):
    assert d.dispatch(None) is None
    assert d.dispatch(Action("none")).output == ""


@pytest.mark.parametrize("args", [("shell", "rm -rf /"), ("speak", ""), ("phrase_add", "  ")])
def test_invalid_actions_rejected(args):
    with pytest.raises(ValueError):
        Action(*args)


def test_phrase_buffer_limits_and_punctuation():
    p = PhraseBuffer(max_words=3)
    for w in "one two three four".split():
        p.add(w)
    assert p.words == ["two", "three", "four"]
    p.clear()
    p.add("are you okay?")
    assert p.text == "Are you okay?"
