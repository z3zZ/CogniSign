"""On-screen feedback: skeleton, current gesture, hold ring, bindings, sentence, toasts.

Clear feedback matters more than usual here: the person needs to know the
system has seen them *before* it speaks for them, and needs to see what it
did afterwards. Every action produces a toast and a coloured border flash;
the bindings panel shows what every gesture does in the current mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .tracker import Hand

# BGR
ACCENT = (241, 196, 72)     # cyan-blue
FIRE = (120, 230, 120)      # green: something happened
WARN = (60, 170, 255)       # amber: nothing to do / paused
SENTENCE = (230, 140, 200)  # violet: sentence mode
MUTED = (165, 165, 165)
DIM = (110, 110, 110)
WHITE = (245, 245, 245)
PANEL = (24, 24, 28)

KIND_COLOUR = {"said": FIRE, "added": SENTENCE, "info": ACCENT, "warn": WARN}

FONT = cv2.FONT_HERSHEY_DUPLEX
SMALL = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

PANEL_WIDTH = 330


@dataclass
class HudState:
    label: str | None = None
    source: str = "none"
    confidence: float = 0.0
    progress: float = 0.0
    mode: str = "quick"
    phrase_words: list[str] = field(default_factory=list)
    empty_hint: str = ""
    toast: str = ""
    toast_kind: str = "info"
    toast_alpha: float = 0.0     # 0..1
    flash: float = 0.0           # 0..1 border flash after an action
    flash_kind: str = "said"
    paused: bool = False
    fps: float = 0.0
    bindings: list[tuple[str, str]] = field(default_factory=list)
    show_bindings: bool = True
    fired_label: str | None = None
    keys_hint: str = "q quit  p pause  m mode  c clear  b bindings"


def pretty(name: str) -> str:
    return name.replace("_", " ")


def _panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           alpha: float = 0.7, colour=PANEL) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        overlay = np.full_like(roi, colour)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _text(frame, text, org, scale=0.6, colour=WHITE, font=SMALL, thick=1) -> int:
    cv2.putText(frame, text, org, font, scale, colour, thick, cv2.LINE_AA)
    return cv2.getTextSize(text, font, scale, thick)[0][0]


def _fit(text: str, max_px: int, scale: float, font=SMALL, keep_end: bool = False) -> str:
    """Trim text with an ellipsis so it fits in max_px."""
    if cv2.getTextSize(text, font, scale, 1)[0][0] <= max_px:
        return text
    while text and cv2.getTextSize("..." + text if keep_end else text + "...",
                                   font, scale, 1)[0][0] > max_px:
        text = text[1:] if keep_end else text[:-1]
    return ("..." + text) if keep_end else (text + "...")


def draw_hand(frame: np.ndarray, hand: Hand, colour=ACCENT) -> None:
    pts = hand.pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), colour, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, tuple(map(int, p)), r, WHITE, -1, cv2.LINE_AA)


def draw_ring(frame: np.ndarray, centre: tuple[int, int], progress: float, colour=ACCENT) -> None:
    cv2.circle(frame, centre, 28, DIM, 2, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, centre, (28, 28), -90, 0, int(360 * progress), colour, 5, cv2.LINE_AA)


def _badge(frame, text, x, y, colour) -> int:
    (tw, th), _ = cv2.getTextSize(text, SMALL, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 16, y + 6), colour, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 8, y), SMALL, 0.5, PANEL, 1, cv2.LINE_AA)
    return tw + 16


def draw_top(frame: np.ndarray, state: HudState) -> None:
    w = frame.shape[1]
    _panel(frame, 0, 0, w, 76)
    if state.paused:
        colour, title = WARN, "PAUSED - make the pause gesture to resume"
    else:
        colour = KIND_COLOUR.get(state.flash_kind, FIRE) if state.flash > 0 else ACCENT
        title = pretty(state.label) if state.label else "show a gesture"
    draw_ring(frame, (40, 38), 1.0 if state.flash > 0 else state.progress, colour)
    _text(frame, _fit(title, w - 360, 0.9, FONT), (84, 34), 0.9, WHITE, FONT)
    if state.label and not state.paused:
        detail = f"{state.source}  {state.confidence:.0%}  -  hold to fire"
    else:
        detail = "no gesture"
    _text(frame, detail, (84, 62), 0.5, MUTED)

    mode_colour = SENTENCE if state.mode == "sentence" else ACCENT
    label = "SENTENCE MODE" if state.mode == "sentence" else "QUICK MODE"
    bw = cv2.getTextSize(label, SMALL, 0.5, 1)[0][0] + 16
    _badge(frame, label, w - bw - 16, 34, mode_colour)
    _text(frame, f"{state.fps:3.0f} fps", (w - 80, 62), 0.45, DIM)


def draw_bindings(frame: np.ndarray, state: HudState) -> None:
    h, w = frame.shape[:2]
    if not state.show_bindings or not state.bindings:
        return
    x1, y1 = w - PANEL_WIDTH, 86
    row_h = 30
    y2 = min(h - 102, y1 + 44 + row_h * len(state.bindings))
    _panel(frame, x1, y1, w - 10, y2, alpha=0.72)
    _text(frame, "GESTURES", (x1 + 14, y1 + 26), 0.5, MUTED)
    _text(frame, "b to hide", (w - 100, y1 + 26), 0.4, DIM)
    y = y1 + 44
    for name, effect in state.bindings:
        if y + row_h > y2 + 1:
            break
        active = name == state.label and not state.paused
        fired = name == state.fired_label and state.flash > 0
        if fired:
            _panel(frame, x1, y, w - 10, y + row_h - 2, 0.55,
                   KIND_COLOUR.get(state.flash_kind, FIRE))
        elif active:
            fill = x1 + int((w - 10 - x1) * state.progress)
            _panel(frame, x1, y, fill, y + row_h - 2, 0.45, ACCENT)
            cv2.rectangle(frame, (x1, y), (x1 + 4, y + row_h - 2), ACCENT, -1)
        name_col = WHITE if (active or fired) else MUTED
        _text(frame, _fit(pretty(name), 120, 0.5), (x1 + 14, y + 20), 0.5, name_col)
        _text(frame, _fit(effect, PANEL_WIDTH - 160, 0.48), (x1 + 140, y + 20), 0.48,
              WHITE if (active or fired) else DIM)
        y += row_h


def draw_sentence(frame: np.ndarray, state: HudState) -> None:
    h, w = frame.shape[:2]
    _panel(frame, 0, h - 92, w, h)
    colour = SENTENCE if state.mode == "sentence" else ACCENT
    cv2.rectangle(frame, (0, h - 92), (5, h), colour, -1)

    x, y = 22, h - 52
    if not state.phrase_words:
        _text(frame, _fit(state.empty_hint, w - 60, 0.7, FONT), (x, y), 0.7, DIM, FONT)
    else:
        # show the newest words that fit; last word highlighted
        words = state.phrase_words
        max_px = w - 60
        shown: list[str] = []
        used = 0
        for word in reversed(words):
            ww = cv2.getTextSize(word + " ", FONT, 0.8, 1)[0][0]
            if used + ww > max_px - 40:
                break
            shown.insert(0, word)
            used += ww
        if len(shown) < len(words):
            x += _text(frame, "... ", (x, y), 0.8, DIM, FONT)
        for i, word in enumerate(shown):
            last = i == len(shown) - 1
            x += _text(frame, word + " ", (x, y), 0.8, colour if last else WHITE, FONT)
        if int(cv2.getTickCount() / cv2.getTickFrequency() * 2) % 2 == 0:
            cv2.line(frame, (x, y - 24), (x, y + 6), colour, 2)
    _text(frame, state.keys_hint, (22, h - 18), 0.45, DIM)


def draw_toast(frame: np.ndarray, state: HudState) -> None:
    if not state.toast or state.toast_alpha <= 0:
        return
    h, w = frame.shape[:2]
    usable = w - (PANEL_WIDTH if state.show_bindings and state.bindings else 0)
    text = _fit(state.toast, usable - 120, 1.0, FONT)
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    cx = usable // 2
    x1, x2 = cx - tw // 2 - 28, cx + tw // 2 + 28
    y2 = h - 112
    y1 = y2 - th - 36
    colour = KIND_COLOUR.get(state.toast_kind, ACCENT)
    a = max(0.0, min(1.0, state.toast_alpha))
    _panel(frame, x1, y1, x2, y2, 0.85 * a)
    roi = frame[max(y1, 0):y2, max(x1, 0):x2]
    if roi.size:
        layer = roi.copy()
        cv2.rectangle(layer, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), colour, 3)
        cv2.rectangle(layer, (0, 0), (6, roi.shape[0] - 1), colour, -1)
        cv2.putText(layer, text, (28, roi.shape[0] - 20), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, a, roi, 1 - a, 0, dst=roi)


def draw_flash(frame: np.ndarray, state: HudState) -> None:
    if state.flash <= 0:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), KIND_COLOUR.get(state.flash_kind, FIRE), 8)


def draw(frame: np.ndarray, hands: list[Hand], state: HudState) -> np.ndarray:
    if state.paused:
        colour = WARN
    elif state.flash > 0:
        colour = KIND_COLOUR.get(state.flash_kind, FIRE)
    else:
        colour = ACCENT
    for hand in hands:
        draw_hand(frame, hand, colour)
    draw_top(frame, state)
    draw_bindings(frame, state)
    draw_sentence(frame, state)
    draw_toast(frame, state)
    draw_flash(frame, state)
    return frame
