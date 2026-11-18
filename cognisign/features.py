"""Turn raw hand landmarks into features that don't care where your hand is.

MediaPipe gives 21 (x, y, z) points per hand. The same gesture produces very
different raw numbers depending on where the hand is in frame, how close it is
to the camera and how it's tilted. Before anything is compared we:

1. move the wrist to the origin            -> position invariant
2. divide by palm length (wrist -> middle MCP) -> distance/scale invariant
3. mirror left hands onto right hands      -> one template works for both
4. rotate so the palm points "up"          -> tilt invariant shape

Rotation matters for some gestures (thumbs up vs thumbs down are the same
shape, upside down), so the palm direction is kept as a separate, weighted
orientation component instead of being thrown away.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
FEATURE_VERSION = 2
ORIENTATION_WEIGHT = 1.5

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

_EPS = 1e-6


class InvalidLandmarks(ValueError):
    """Raised when landmarks can't be turned into features."""


def to_array(landmarks) -> np.ndarray:
    """Accept a (21, 3) / (21, 2) array, a flat list, or MediaPipe landmark objects."""
    if len(landmarks) and hasattr(landmarks[0], "x"):
        pts = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in landmarks], dtype=float)
    else:
        pts = np.asarray(landmarks, dtype=float)
    if pts.ndim == 1:
        if pts.size == NUM_LANDMARKS * 3:
            pts = pts.reshape(NUM_LANDMARKS, 3)
        elif pts.size == NUM_LANDMARKS * 2:
            pts = pts.reshape(NUM_LANDMARKS, 2)
    if pts.shape[0] != NUM_LANDMARKS or pts.shape[1] not in (2, 3):
        raise InvalidLandmarks(f"expected 21 landmarks, got array of shape {pts.shape}")
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((NUM_LANDMARKS, 1))])
    if not np.all(np.isfinite(pts)):
        raise InvalidLandmarks("landmarks contain NaN or inf")
    return pts


def palm_size(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2]))


def centre_and_scale(landmarks, handedness: str | None = None) -> np.ndarray:
    """Wrist at origin, palm length 1, left hands mirrored. Orientation untouched."""
    pts = to_array(landmarks).copy()
    pts -= pts[WRIST]
    size = palm_size(pts)
    if size < _EPS:
        raise InvalidLandmarks("hand is degenerate (palm length ~0)")
    pts /= size
    if handedness and handedness.lower().startswith("l"):
        pts[:, 0] *= -1.0
    return pts


def palm_direction(pts: np.ndarray) -> np.ndarray:
    """Unit vector from wrist to middle-finger MCP in the image plane."""
    v = pts[MIDDLE_MCP, :2] - pts[WRIST, :2]
    return v / max(np.linalg.norm(v), _EPS)


def canonicalise(landmarks, handedness: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (shape, orientation).

    shape:       (21, 3) points with the palm rotated to point straight up (-y).
    orientation: unit 2-vector of where the palm was actually pointing.
    """
    pts = centre_and_scale(landmarks, handedness)
    direction = palm_direction(pts)
    # rotate so `direction` lands on (0, -1); image y grows downwards
    phi = np.arctan2(-1.0, 0.0) - np.arctan2(direction[1], direction[0])
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    shape = pts.copy()
    shape[:, :2] = pts[:, :2] @ rot.T
    return shape, direction


def extract_features(landmarks, handedness: str | None = None) -> np.ndarray:
    """Flat feature vector: 63 shape values + 2 weighted orientation values."""
    shape, direction = canonicalise(landmarks, handedness)
    return np.concatenate([shape.ravel(), ORIENTATION_WEIGHT * direction])


FEATURE_SIZE = NUM_LANDMARKS * 3 + 2
