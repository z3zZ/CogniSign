"""Teach CogniSign a new gesture from your own hand in a few seconds."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np

from .classifier import GestureLibrary
from .features import extract_features
from .tracker import Hand, primary_hand


class SampleCollector:
    """Collects spaced-out samples of one gesture while a hand is visible.

    Spacing samples out (rather than grabbing 30 near-identical frames in one
    second) gives the classifier natural variation in angle and distance,
    which is what makes it robust later.
    """

    def __init__(self, label: str, target: int = 40, interval_ms: int = 80,
                 countdown_ms: int = 3000):
        self.label = label
        self.target = target
        self.interval_ms = interval_ms
        self.countdown_ms = countdown_ms
        self.samples: list[np.ndarray] = []
        self._start: int | None = None
        self._last = -(10**12)

    def phase(self, now_ms: int) -> str:
        if self._start is None:
            return "waiting"
        if now_ms - self._start < self.countdown_ms:
            return "countdown"
        return "done" if self.done else "recording"

    def countdown_left(self, now_ms: int) -> float:
        if self._start is None:
            return self.countdown_ms / 1000
        return max(0.0, (self.countdown_ms - (now_ms - self._start)) / 1000)

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.target

    def update(self, hand: Hand | None, now_ms: int) -> bool:
        """Feed a frame. Returns True when a sample was captured."""
        if self._start is None:
            self._start = now_ms
        if self.phase(now_ms) != "recording" or hand is None:
            return False
        if now_ms - self._last < self.interval_ms:
            return False
        self.samples.append(extract_features(hand.landmarks, hand.handedness))
        self._last = now_ms
        return True

    def commit(self, library: GestureLibrary) -> int:
        for s in self.samples:
            library.add(self.label, s)
        return len(self.samples)


def record(label: str, tracker, frames: Iterator[np.ndarray], *, samples: int = 40,
           show: bool = True, clock: Callable[[], float] = time.monotonic) -> list[np.ndarray]:
    import cv2

    from .hud import ACCENT, FIRE, WHITE, draw_hand

    collector = SampleCollector(label, target=samples)
    start = clock()
    try:
        for frame in frames:
            now_ms = int((clock() - start) * 1000)
            hand = primary_hand(tracker.process(frame, now_ms))
            collector.update(hand, now_ms)
            if show:
                phase = collector.phase(now_ms)
                if hand:
                    draw_hand(frame, hand, FIRE if phase == "recording" else ACCENT)
                if phase == "countdown":
                    msg = f"Get ready to show '{label}'... {collector.countdown_left(now_ms):.0f}"
                elif phase == "recording":
                    msg = (f"Recording '{label}' {len(collector.samples)}/{samples}"
                           " - move your hand around a little")
                    if hand is None:
                        msg = "Recording paused: no hand in view"
                else:
                    msg = "Done!"
                cv2.putText(frame, msg, (20, 44), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2,
                            cv2.LINE_AA)
                cv2.imshow("CogniSign - record", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if collector.done:
                break
    finally:
        if show:
            cv2.destroyAllWindows()
    return collector.samples
