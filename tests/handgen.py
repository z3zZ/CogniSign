"""Synthetic hand poses for tests, in MediaPipe landmark order.

Built in a canonical frame (wrist at origin, fingers pointing up = -y, image
y grows downwards), then rotated / scaled / moved like a real hand in frame.
"""

from __future__ import annotations

import numpy as np

_MCP_X = {5: 0.30, 9: 0.10, 13: -0.10, 17: -0.30}


def make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True, *,
              pinch=False, rotate_deg=0.0, scale=0.12, offset=(0.5, 0.6),
              mirror=False, noise=0.0, rng=None) -> np.ndarray:
    pts = np.zeros((21, 3))
    pts[1] = (0.25, -0.25, 0)
    pts[2] = (0.45, -0.45, 0)
    if thumb:
        pts[3] = (0.60, -0.60, 0)
        pts[4] = (0.75, -0.75, 0)
    else:
        pts[3] = (0.35, -0.65, 0)
        pts[4] = (0.15, -0.70, 0)

    for mcp, up in zip((5, 9, 13, 17), (index, middle, ring, pinky), strict=False):
        x = _MCP_X[mcp]
        pts[mcp] = (x, -1.0, 0)
        if up:
            pts[mcp + 1] = (x, -1.45, 0)
            pts[mcp + 2] = (x, -1.75, 0)
            pts[mcp + 3] = (x, -2.00, 0)
        else:
            pts[mcp + 1] = (x, -1.35, -0.1)
            pts[mcp + 2] = (x, -1.20, -0.15)
            pts[mcp + 3] = (x, -0.90, -0.1)

    if pinch:  # index curls down to meet the thumb tip
        pts[6] = (0.36, -1.35, 0)
        pts[7] = (0.52, -1.10, 0)
        pts[8] = pts[4] + (-0.04, -0.04, 0)

    th = np.radians(rotate_deg)
    c, s = np.cos(th), np.sin(th)
    xy = pts[:, :2] @ np.array([[c, -s], [s, c]]).T
    pts[:, :2] = xy
    if mirror:  # a left hand is the mirror image of the same right-hand pose
        pts[:, 0] *= -1
    pts *= scale
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    if noise:
        rng = rng or np.random.default_rng(0)
        pts += rng.normal(0, noise * scale, pts.shape)
    return pts


POSES = {
    "open_palm": dict(),
    "fist": dict(thumb=False, index=False, middle=False, ring=False, pinky=False),
    "point": dict(thumb=False, middle=False, ring=False, pinky=False),
    "peace": dict(thumb=False, ring=False, pinky=False),
    "three": dict(thumb=False, pinky=False),
    "rock": dict(thumb=False, middle=False, ring=False),
    "call_me": dict(index=False, middle=False, ring=False),
    "ok": dict(pinch=True, index=False),
    # thumb-only hand, rotated so the thumb points straight up / down
    "thumbs_up": dict(index=False, middle=False, ring=False, pinky=False, rotate_deg=-45),
    "thumbs_down": dict(index=False, middle=False, ring=False, pinky=False, rotate_deg=135),
}


def pose(name: str, **overrides) -> np.ndarray:
    kw = {**POSES[name], **overrides}
    if "rotate_deg" in POSES[name] and "rotate_deg" in overrides:
        kw["rotate_deg"] = POSES[name]["rotate_deg"] + overrides["rotate_deg"]
    return make_hand(**kw)
