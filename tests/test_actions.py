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


def test_sentence_mode_turns_speak_into_add(d):
    d.dispatch(Action("toggle_mode"))
    res = d.dispatch(Action("speak", "Yes"))
    assert res.action.type == "phrase_add" and res.message == "Added: Yes"
    assert d.speaker.said == [] and d.phrase.words == ["Yes"]
    assert d.describe(Action("speak", "No")) == 'add "No"'
    d.dispatch(Action("toggle_mode"))
    assert d.describe(Action("speak", "No")) == 'say "No"'


@pytest.mark.parametrize("action, message, kind", [
    (Action("speak", "Hi"), "Said: Hi", "said"),
    (Action("phrase_add", "water"), "Added: water", "added"),
    (Action("phrase_speak"), "Sentence is empty, nothing to say", "warn"),
    (Action("phrase_undo"), "Sentence is already empty", "warn"),
    (Action("hotkey", "right"), "Pressed: right", "info"),
])
def test_every_action_reports_a_message(d, action, message, kind):
    res = d.dispatch(action)
    assert (res.message, res.kind) == (message, kind)


def test_undo_reports_removed_word(d):
    d.dispatch(Action("phrase_add", "water"))
    assert d.dispatch(Action("phrase_undo")).message == "Removed: water"
