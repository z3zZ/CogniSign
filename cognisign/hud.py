"""On-screen feedback: skeleton, current gesture, hold ring, sentence bar.

Clear feedback matters more than usual here: the person needs to know the
system has seen them *before* it speaks for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tracker import Hand

# BGR
ACCENT = (241, 196, 72)     # warm cyan-blue
FIRE = (120, 230, 120)
MUTED = (170, 170, 170)
PAUSED = (60, 140, 255)
WHITE = (245, 245, 245)
PANEL = (24, 24, 28)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HudState:
    label: str | None = None
    source: str = "none"
    confidence: float = 0.0
    progress: float = 0.0
    last_output: str = ""
    flash: float = 0.0          # 0..1, fades after an action fires
    phrase: str = ""
    paused: bool = False
    fps: float = 0.0
    hint: str = "q quit  |  p pause  |  c clear sentence"


def _panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.65) -> None:
    roi = frame[max(y1, 0):y2, max(x1, 0):x2]
    if roi.size:
        overlay = np.full_like(roi, PANEL)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_hand(frame: np.ndarray, hand: Hand, colour=ACCENT) -> None:
    pts = hand.pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), colour, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, tuple(map(int, p)), r, WHITE, -1, cv2.LINE_AA)


def draw_ring(frame: np.ndarray, centre: tuple[int, int], progress: float, colour=ACCENT) -> None:
    cv2.circle(frame, centre, 28, MUTED, 2, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, centre, (28, 28), -90, 0, int(360 * progress), colour, 5, cv2.LINE_AA)


def draw(frame: np.ndarray, hands: list[Hand], state: HudState) -> np.ndarray:
    h, w = frame.shape[:2]
    colour = PAUSED if state.paused else (FIRE if state.flash > 0 else ACCENT)
    for hand in hands:
        draw_hand(frame, hand, colour)

    # top bar: gesture + hold ring
    _panel(frame, 0, 0, w, 76)
    title = "PAUSED  (make the pause gesture to resume)" if state.paused else (
        state.label.replace("_", " ") if state.label else "show a gesture")
    cv2.putText(frame, title, (84, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 1, cv2.LINE_AA)
    detail = f"{state.source}  {state.confidence:.0%}" if state.label else "no gesture"
    cv2.putText(frame, detail, (84, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)
    draw_ring(frame, (40, 38), 1.0 if state.flash > 0 else state.progress, colour)
    cv2.putText(frame, f"{state.fps:4.0f} fps", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)

    # bottom bar: sentence builder + last output
    _panel(frame, 0, h - 92, w, h)
    sentence = state.phrase or "sentence builder is empty"
    cv2.putText(frame, sentence[-70:], (20, h - 54), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                WHITE if state.phrase else MUTED, 1, cv2.LINE_AA)
    if state.last_output:
        cv2.putText(frame, f"> {state.last_output[-60:]}", (20, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FIRE, 1, cv2.LINE_AA)
    cv2.putText(frame, state.hint, (w - 360, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                MUTED, 1, cv2.LINE_AA)
    return frame
