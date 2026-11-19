"""What a gesture actually does: speak, type, press keys, or build a sentence.

Output backends are injected so the logic can be tested without a speaker or
keyboard, and so the app keeps working (printing instead of speaking) on a
machine without a TTS engine.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Protocol

from .phrase import PhraseBuffer

log = logging.getLogger(__name__)

ACTION_TYPES = {
    "speak": "Say the text out loud",
    "type": "Type the text into whatever window has focus",
    "hotkey": "Press a key combination, e.g. ctrl+c or media_play_pause",
    "phrase_add": "Add a word to the sentence builder",
    "phrase_speak": "Speak the built sentence and clear it",
    "phrase_undo": "Remove the last word from the sentence",
    "phrase_clear": "Clear the sentence",
    "toggle_pause": "Pause / resume all other gestures",
    "toggle_mode": "Switch between quick replies and sentence building",
    "none": "Do nothing (useful for a 'rest' pose)",
}
NEEDS_VALUE = {"speak", "type", "hotkey", "phrase_add"}


@dataclass(frozen=True)
class Action:
    type: str
    value: str = ""

    def __post_init__(self):
        if self.type not in ACTION_TYPES:
            raise ValueError(f"unknown action type {self.type!r}; "
                             f"choose from {', '.join(sorted(ACTION_TYPES))}")
        if self.type in NEEDS_VALUE and not str(self.value).strip():
            raise ValueError(f"action {self.type!r} needs a value")

    def describe(self) -> str:
        return f"{self.type}: {self.value}" if self.value else self.type


class Speaker(Protocol):
    def say(self, text: str) -> None: ...
    def close(self) -> None: ...


class Keyboard(Protocol):
    def type_text(self, text: str) -> None: ...
    def hotkey(self, combo: str) -> None: ...


class ConsoleSpeaker:
    """Fallback when no TTS engine is available."""

    def say(self, text: str) -> None:
        print(f"[speak] {text}", flush=True)

    def close(self) -> None:
        pass


class Pyttsx3Speaker:
    """Offline text-to-speech on a background thread so the camera never stalls."""

    def __init__(self, rate: int = 170, voice: str | None = None):
        import pyttsx3  # imported lazily: optional at runtime

        self._q: queue.Queue[str | None] = queue.Queue()
        self._rate, self._voice = rate, voice
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, args=(pyttsx3,), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._error:
            raise self._error

    def _run(self, pyttsx3) -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            if self._voice:
                engine.setProperty("voice", self._voice)
        except Exception as exc:  # pragma: no cover - depends on OS voices
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        while (text := self._q.get()) is not None:
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str) -> None:
        self._q.put(text)

    def close(self) -> None:
        self._q.put(None)


class NullKeyboard:
    def type_text(self, text: str) -> None:
        log.warning("keyboard output unavailable, would have typed %r", text)

    def hotkey(self, combo: str) -> None:
        log.warning("keyboard output unavailable, would have pressed %r", combo)


class PynputKeyboard:
    def __init__(self):
        from pynput.keyboard import Controller, Key  # lazy, optional

        self._kb = Controller()
        self._Key = Key

    def type_text(self, text: str) -> None:
        self._kb.type(text)

    def _key(self, name: str):
        name = name.strip().lower()
        if len(name) == 1:
            return name
        aliases = {"control": "ctrl", "return": "enter", "escape": "esc", "win": "cmd"}
        name = aliases.get(name, name)
        try:
            return getattr(self._Key, name)
        except AttributeError as exc:
            raise ValueError(f"unknown key {name!r}") from exc

    def hotkey(self, combo: str) -> None:
        keys = [self._key(k) for k in combo.split("+")]
        for k in keys:
            self._kb.press(k)
        for k in reversed(keys):
            self._kb.release(k)


MODES = ("quick", "sentence")


@dataclass
class DispatchResult:
    action: Action          # the action that actually ran (after mode translation)
    output: str = ""        # machine-friendly result, e.g. the spoken text
    message: str = ""       # what to show the person on screen
    kind: str = "info"      # said | added | info | warn


class ActionDispatcher:
    """Runs actions and reports back in words a person can read at a glance.

    Two modes:
      quick     gestures do exactly what they're mapped to ("Yes", "No", ...)
      sentence  gestures mapped to `speak` add their words to the sentence
                instead, so the same vocabulary builds full sentences
    """

    def __init__(self, speaker: Speaker | None = None, keyboard: Keyboard | None = None,
                 phrase: PhraseBuffer | None = None):
        self.speaker = speaker or ConsoleSpeaker()
        self.keyboard = keyboard or NullKeyboard()
        self.phrase = phrase or PhraseBuffer()
        self.paused = False
        self.mode = "quick"

    def effective(self, action: Action | None) -> Action | None:
        """The action as it will run in the current mode."""
        if action is not None and self.mode == "sentence" and action.type == "speak":
            return Action("phrase_add", action.value)
        return action

    def describe(self, action: Action | None) -> str:
        """Short label for the bindings panel."""
        a = self.effective(action)
        if a is None:
            return "(not set)"
        return {
            "speak": f'say "{a.value}"',
            "type": f'type "{a.value}"',
            "hotkey": f"press {a.value}",
            "phrase_add": f'add "{a.value}"',
            "phrase_speak": "speak sentence",
            "phrase_undo": "undo last word",
            "phrase_clear": "clear sentence",
            "toggle_pause": "pause / resume",
            "toggle_mode": "quick <-> sentence",
            "none": "nothing",
        }[a.type]

    def dispatch(self, action: Action | None) -> DispatchResult | None:
        action = self.effective(action)
        if action is None:
            return None
        if self.paused and action.type != "toggle_pause":
            return None
        handler = getattr(self, f"_do_{action.type}")
        out, message, kind = handler(action.value)
        log.info("gesture action %s -> %s", action.describe(), out)
        return DispatchResult(action, out, message, kind)

    def _do_speak(self, value: str):
        self.speaker.say(value)
        return value, f"Said: {value}", "said"

    def _do_type(self, value: str):
        self.keyboard.type_text(value)
        return value, f"Typed: {value}", "info"

    def _do_hotkey(self, value: str):
        self.keyboard.hotkey(value)
        return value, f"Pressed: {value}", "info"

    def _do_phrase_add(self, value: str):
        self.phrase.add(value)
        return self.phrase.text, f"Added: {value}", "added"

    def _do_phrase_speak(self, _value: str):
        text = self.phrase.take()
        if not text:
            return "", "Sentence is empty, nothing to say", "warn"
        self.speaker.say(text)
        return text, f"Said: {text}", "said"

    def _do_phrase_undo(self, _value: str):
        word = self.phrase.undo()
        if word is None:
            return "", "Sentence is already empty", "warn"
        return self.phrase.text, f"Removed: {word}", "info"

    def _do_phrase_clear(self, _value: str):
        had = len(self.phrase)
        self.phrase.clear()
        return "", "Sentence cleared" if had else "Sentence is already empty", "info"

    def _do_toggle_pause(self, _value: str):
        self.paused = not self.paused
        return ("paused", "Paused", "warn") if self.paused else ("resumed", "Resumed", "info")

    def _do_toggle_mode(self, _value: str):
        self.mode = "sentence" if self.mode == "quick" else "quick"
        if self.mode == "sentence":
            return self.mode, "Sentence mode: gestures now add words", "info"
        return self.mode, "Quick mode: gestures speak straight away", "info"

    def _do_none(self, _value: str):
        return "", "", "info"

    def close(self) -> None:
        self.speaker.close()


def make_speaker(rate: int = 170, voice: str | None = None) -> Speaker:
    try:
        return Pyttsx3Speaker(rate=rate, voice=voice)
    except Exception as exc:
        log.warning("text-to-speech unavailable (%s); printing instead", exc)
        return ConsoleSpeaker()


def make_keyboard() -> Keyboard:
    try:
        return PynputKeyboard()
    except Exception as exc:
        log.warning("keyboard control unavailable (%s)", exc)
        return NullKeyboard()
