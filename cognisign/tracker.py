"""Hand tracking with Google's MediaPipe Hand Landmarker.

MediaPipe finds up to two hands per frame and returns 21 3D landmarks for
each, fast enough to run on a laptop CPU in real time. The model file
(~7.5 MB) is downloaded once on first run and cached in ~/.cognisign/models.
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import models_dir

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_NAME = "hand_landmarker.task"
MIN_MODEL_BYTES = 1_000_000


class ModelDownloadError(RuntimeError):
    pass


@dataclass
class Hand:
    landmarks: np.ndarray        # (21, 3), x/y in aspect-corrected image units
    pixels: np.ndarray           # (21, 2) int pixel coords for drawing
    handedness: str              # "Left" / "Right"
    score: float


def ensure_model(directory: Path | None = None, url: str = MODEL_URL) -> Path:
    directory = Path(directory or models_dir())
    path = directory / MODEL_NAME
    if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmark model (first run only)...")
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, os.fdopen(fd, "wb") as out:
            while chunk := resp.read(1 << 16):
                out.write(chunk)
        if os.path.getsize(tmp) < MIN_MODEL_BYTES:
            raise ModelDownloadError("downloaded model is too small, the download probably failed")
        os.replace(tmp, path)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        if isinstance(exc, ModelDownloadError):
            raise
        raise ModelDownloadError(
            f"couldn't download the hand model ({exc}). Download it manually from\n"
            f"  {url}\nand save it as {path}"
        ) from exc
    return path


def to_hands(result, width: int, height: int) -> list[Hand]:
    """Convert a MediaPipe HandLandmarkerResult into :class:`Hand` objects.

    MediaPipe normalises x by width and y by height, so on a 16:9 frame one
    unit of x is longer than one unit of y. We rescale x by width/height so
    distances and angles are true to life before any geometry is done.
    """
    hands: list[Hand] = []
    aspect = width / height if height else 1.0
    for lms, handed in zip(result.hand_landmarks, result.handedness, strict=False):
        raw = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
        corrected = raw.copy()
        corrected[:, 0] *= aspect
        corrected[:, 2] *= aspect
        pixels = np.column_stack([raw[:, 0] * width, raw[:, 1] * height]).astype(int)
        cat = handed[0] if handed else None
        hands.append(Hand(
            landmarks=corrected,
            pixels=pixels,
            handedness=getattr(cat, "category_name", "Right") or "Right",
            score=float(getattr(cat, "score", 1.0)),
        ))
    return hands


class HandTracker:
    def __init__(self, model_path: str | os.PathLike | None = None, num_hands: int = 1,
                 min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        # VIDEO mode requires strictly increasing timestamps
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        h, w = frame_bgr.shape[:2]
        return to_hands(result, w, h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def primary_hand(hands: list[Hand]) -> Hand | None:
    return max(hands, key=lambda h: h.score) if hands else None
