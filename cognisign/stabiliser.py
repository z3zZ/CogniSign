"""Turn a noisy stream of per-frame guesses into deliberate, one-off events.

At 30 fps a hand passing through a pose for a split second would otherwise
fire it. The stabiliser only fires when a gesture is *held* for ``hold_ms``,
fires it once, and won't fire it again until the hand changes pose (so holding
"yes" says "yes" once, not thirty times). Short tracking dropouts inside a
hold are forgiven so the progress ring doesn't reset on a single bad frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GestureEvent:
    label: str
    timestamp_ms: int


class GestureStabiliser:
    def __init__(self, hold_ms: int = 600, cooldown_ms: int = 700, grace_ms: int = 150):
        if hold_ms < 0 or cooldown_ms < 0 or grace_ms < 0:
            raise ValueError("timings must be non-negative")
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.grace_ms = grace_ms
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._since = 0
        self._last_seen = 0
        self._armed = True
        self._last_fire = -(10**12)
        self._now = 0

    @property
    def candidate(self) -> str | None:
        return self._candidate

    @property
    def progress(self) -> float:
        """0..1 hold progress for the current candidate (drives the HUD ring)."""
        if self._candidate is None or not self._armed:
            return 0.0
        if self.hold_ms == 0:
            return 1.0
        return max(0.0, min(1.0, (self._now - self._since) / self.hold_ms))

    @property
    def armed(self) -> bool:
        return self._armed

    def update(self, label: str | None, timestamp_ms: int) -> GestureEvent | None:
        now = int(timestamp_ms)
        self._now = now

        if label is not None and label == self._candidate:
            self._last_seen = now
        elif self._candidate is not None and now - self._last_seen <= self.grace_ms:
            # brief flicker to nothing / something else: keep holding
            return None
        else:
            # genuinely new pose (or hand gone): start over and re-arm
            self._candidate = label
            self._since = now
            self._last_seen = now
            self._armed = True
            return None

        if (
            self._armed
            and now - self._since >= self.hold_ms
            and now - self._last_fire >= self.cooldown_ms
        ):
            self._armed = False
            self._last_fire = now
            return GestureEvent(label, now)
        return None
