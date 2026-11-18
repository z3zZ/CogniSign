"""Built-in gestures that work with zero training.

These are geometric rules on top of the landmarks, so CogniSign is useful the
second it opens. Anything more personal (or anything these rules can't see)
is learned from your own recordings by :mod:`cognisign.classifier`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import (
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    centre_and_scale,
)

# (pip, tip) for index, middle, ring, pinky
_FINGER_JOINTS = ((6, 8), (10, 12), (14, 16), (18, 20))

EXTENSION_RATIO = 1.12   # tip must be this much further from the wrist than the PIP
THUMB_RATIO = 1.0        # thumb tip vs thumb MCP distance from the pinky knuckle
PINCH_DISTANCE = 0.35    # thumb tip <-> index tip, in palm lengths
VERTICAL = 0.7           # |cos| threshold for "pointing up / down"

PRESET_GESTURES: dict[str, str] = {
    "open_palm": "All five fingers spread",
    "fist": "Closed fist",
    "thumbs_up": "Thumb up, fingers curled",
    "thumbs_down": "Thumb down, fingers curled",
    "point": "Index finger only",
    "peace": "Index + middle (V sign)",
    "three": "Index + middle + ring",
    "rock": "Index + pinky (horns)",
    "call_me": "Thumb + pinky (phone)",
    "ok": "Thumb and index touching, other fingers up",
}


@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


def finger_states(pts: np.ndarray) -> FingerState:
    """Which fingers are extended. Uses distances only, so tilt doesn't matter."""
    wrist = pts[WRIST, :2]
    fingers = []
    for pip, tip in _FINGER_JOINTS:
        d_tip = np.linalg.norm(pts[tip, :2] - wrist)
        d_pip = np.linalg.norm(pts[pip, :2] - wrist)
        fingers.append(bool(d_tip > d_pip * EXTENSION_RATIO))
    pinky_knuckle = pts[PINKY_MCP, :2]
    d_thumb_tip = np.linalg.norm(pts[THUMB_TIP, :2] - pinky_knuckle)
    d_thumb_mcp = np.linalg.norm(pts[THUMB_MCP, :2] - pinky_knuckle)
    thumb = bool(d_thumb_tip > d_thumb_mcp * THUMB_RATIO)
    return FingerState(thumb, *fingers)


def _thumb_vertical(pts: np.ndarray) -> int:
    """+1 thumb points up, -1 down, 0 sideways."""
    v = pts[THUMB_TIP, :2] - pts[THUMB_MCP, :2]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0
    cos_y = v[1] / n  # image y grows downwards
    if cos_y < -VERTICAL:
        return 1
    if cos_y > VERTICAL:
        return -1
    return 0


def classify_preset(landmarks, handedness: str | None = None) -> str | None:
    """Return a preset gesture name, or None if the pose isn't one of them."""
    pts = centre_and_scale(landmarks, handedness)
    s = finger_states(pts)
    thumb, index, middle, ring, pinky = s.as_tuple()

    pinch = np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]) < PINCH_DISTANCE
    if pinch and middle and ring and pinky:
        return "ok"
    if all(s.as_tuple()):
        return "open_palm"
    if not any(s.as_tuple()):
        return "fist"
    if thumb and pinky and not (index or middle or ring):
        return "call_me"
    if index and pinky and not (middle or ring):
        return "rock"
    if index and middle and not (ring or pinky):
        return "peace"
    if index and middle and ring and not pinky:
        return "three"
    if index and not (middle or ring or pinky or thumb):
        return "point"
    if thumb and not (index or middle or ring or pinky):
        direction = _thumb_vertical(pts)
        if direction == 1:
            return "thumbs_up"
        if direction == -1:
            return "thumbs_down"
    return None


def palm_facing_up(landmarks) -> bool:
    """Handy for HUD hints: is the palm roughly upright in the frame?"""
    pts = centre_and_scale(landmarks)
    v = pts[MIDDLE_MCP, :2] - pts[WRIST, :2]
    return bool(v[1] / max(np.linalg.norm(v), 1e-6) < -VERTICAL)


__all__ = [
    "PRESET_GESTURES",
    "FingerState",
    "classify_preset",
    "finger_states",
    "palm_facing_up",
]
