"""The live loop: camera -> hand tracking -> gesture -> action -> HUD."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .actions import Action, ActionDispatcher, DispatchResult
from .classifier import GestureLibrary, Prediction, Recogniser
from .config import Config, gestures_path
from .hud import HudState, pretty
from .presets import PRESET_GESTURES
from .stabiliser import GestureStabiliser
from .tracker import Hand, primary_hand

log = logging.getLogger(__name__)

FLASH_MS = 500
TOAST_MS = 1800
TOAST_FADE_MS = 400


@dataclass
class FrameResult:
    hands: list[Hand]
    prediction: Prediction | None
    fired: DispatchResult | None


class GestureEngine:
    """Everything except the camera and the window, so it can be tested headless."""

    def __init__(self, config: Config, dispatcher: ActionDispatcher,
                 library: GestureLibrary | None = None):
        rc = config.recognition
        self.config = config
        self.dispatcher = dispatcher
        self.recogniser = Recogniser(
            library if library is not None else GestureLibrary.load(gestures_path()),
            k=rc.k, max_distance=rc.max_distance,
            min_confidence=rc.min_confidence, use_presets=rc.use_presets,
        )
        self.stabiliser = GestureStabiliser(hold_ms=rc.hold_ms, cooldown_ms=rc.cooldown_ms)
        self.hud = HudState()
        self._flash_until = 0
        self._toast_until = 0
        self._now = 0

    # -- helpers -------------------------------------------------------------
    def _bound_to(self, action_type: str) -> str | None:
        for name, action in self.config.gestures.items():
            if action.type == action_type:
                return name
        return None

    def bindings(self) -> list[tuple[str, str]]:
        """(gesture, what it does right now) for every gesture CogniSign knows."""
        names = [g for g in PRESET_GESTURES if g in self.config.gestures]
        names += sorted(set(self.config.gestures) - set(names))
        names += [g for g in self.recogniser.library.labels if g not in names]
        # words first, then sentence controls, then mode / pause, then unbound
        rank = {"phrase_speak": 1, "phrase_undo": 1, "phrase_clear": 1,
                "toggle_mode": 2, "toggle_pause": 3, "none": 4}

        def order(name: str) -> int:
            action = self.config.gestures.get(name)
            return 4 if action is None else rank.get(action.type, 0)

        names.sort(key=order)  # stable: keeps preset order within each group
        return [(n, self.dispatcher.describe(self.config.gestures.get(n))) for n in names]

    def empty_hint(self) -> str:
        if self.dispatcher.mode == "sentence":
            speak = self._bound_to("phrase_speak")
            tail = f", {pretty(speak)} to speak it" if speak else ""
            return f"Sentence mode: gestures add words{tail}"
        mode = self._bound_to("toggle_mode")
        if mode:
            return f"Make the {pretty(mode)} sign to start building a sentence"
        return "Sentence builder is empty"

    def notify(self, message: str, kind: str = "info", fired_label: str | None = None) -> None:
        if not message:
            return
        self.hud.toast, self.hud.toast_kind = message, kind
        self.hud.flash_kind = kind
        self.hud.fired_label = fired_label
        self._toast_until = self._now + TOAST_MS
        self._flash_until = self._now + FLASH_MS

    def manual(self, action: Action) -> DispatchResult | None:
        """Keyboard shortcuts go through the same path as gestures."""
        res = self.dispatcher.dispatch(action)
        if res is not None:
            self.notify(res.message, res.kind)
        self._refresh_hud()
        return res

    def toggle_bindings(self) -> None:
        self.hud.show_bindings = not self.hud.show_bindings

    # -- per frame -----------------------------------------------------------
    def step(self, hands: list[Hand], now_ms: int) -> FrameResult:
        self._now = now_ms
        hand = primary_hand(hands)
        pred = self.recogniser.predict(hand.landmarks, hand.handedness) if hand else None
        label = pred.label if pred else None
        event = self.stabiliser.update(label, now_ms)

        fired = None
        if event is not None:
            action = self.config.action_for(event.label)
            if action is None:
                self.notify(f"{pretty(event.label)} isn't bound to anything", "warn", event.label)
            else:
                fired = self.dispatcher.dispatch(action)
                if fired is not None:
                    self.notify(fired.message, fired.kind, event.label)

        self.hud.label = label
        self.hud.source = pred.source if pred else "none"
        self.hud.confidence = pred.confidence if pred else 0.0
        self.hud.progress = self.stabiliser.progress
        self._refresh_hud()
        return FrameResult(hands, pred, fired)

    def _refresh_hud(self) -> None:
        now = self._now
        self.hud.flash = 1.0 if now < self._flash_until else 0.0
        left = self._toast_until - now
        self.hud.toast_alpha = 0.0 if left <= 0 else min(1.0, left / TOAST_FADE_MS)
        self.hud.phrase_words = self.dispatcher.phrase.words
        self.hud.mode = self.dispatcher.mode
        self.hud.paused = self.dispatcher.paused
        self.hud.bindings = self.bindings()
        self.hud.empty_hint = self.empty_hint()


def camera_frames(index: int, width: int, height: int, mirror: bool) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open camera {index}; try `cognisign run --camera 1`")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("camera stopped returning frames")
                return
            yield cv2.flip(frame, 1) if mirror else frame
    finally:
        cap.release()


def run(config: Config, engine: GestureEngine, tracker,
        frames: Iterator[np.ndarray], *, show: bool = True,
        clock: Callable[[], float] = time.monotonic, max_frames: int | None = None) -> int:
    """Main loop. Returns the number of frames processed."""
    import cv2

    from .hud import draw

    window = "CogniSign"
    start = clock()
    last = start
    fps = 0.0
    n = 0
    try:
        for frame in frames:
            now = clock()
            now_ms = int((now - start) * 1000)
            hands = tracker.process(frame, now_ms)
            engine.step(hands, now_ms)

            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            engine.hud.fps = fps

            n += 1
            if show:
                cv2.imshow(window, draw(frame, hands, engine.hud))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    engine.manual(Action("toggle_pause"))
                elif key == ord("c"):
                    engine.manual(Action("phrase_clear"))
                elif key == ord("m"):
                    engine.manual(Action("toggle_mode"))
                elif key == ord("b"):
                    engine.toggle_bindings()
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if show:
            cv2.destroyAllWindows()
    return n
