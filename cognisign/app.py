"""The live loop: camera -> hand tracking -> gesture -> action -> HUD."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .actions import ActionDispatcher, DispatchResult
from .classifier import GestureLibrary, Prediction, Recogniser
from .config import Config, gestures_path
from .hud import HudState
from .stabiliser import GestureStabiliser
from .tracker import Hand, primary_hand

log = logging.getLogger(__name__)


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

    def step(self, hands: list[Hand], now_ms: int) -> FrameResult:
        hand = primary_hand(hands)
        pred = self.recogniser.predict(hand.landmarks, hand.handedness) if hand else None
        label = pred.label if pred else None
        event = self.stabiliser.update(label, now_ms)

        fired = None
        if event is not None:
            fired = self.dispatcher.dispatch(self.config.action_for(event.label))
            if fired is not None:
                self._flash_until = now_ms + 450
                self.hud.last_output = fired.output or fired.action.describe()

        self.hud.label = label
        self.hud.source = pred.source if pred else "none"
        self.hud.confidence = pred.confidence if pred else 0.0
        self.hud.progress = self.stabiliser.progress
        self.hud.flash = 1.0 if now_ms < self._flash_until else 0.0
        self.hud.phrase = self.dispatcher.phrase.text
        self.hud.paused = self.dispatcher.paused
        return FrameResult(hands, pred, fired)


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
                    engine.dispatcher.paused = not engine.dispatcher.paused
                if key == ord("c"):
                    engine.dispatcher.phrase.clear()
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if show:
            cv2.destroyAllWindows()
    return n
