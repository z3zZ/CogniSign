"""Personal gestures: learn from a handful of your own examples.

Everyone's hands are different, and people with limited mobility often can't
make "textbook" signs. So instead of a fixed model, CogniSign keeps a small
library of *your* recordings and matches new poses against them with a
k-nearest-neighbour vote. Recording a new gesture takes ~5 seconds and there's
no training step, so the library can grow with the person using it.

Anything too far from every recording is rejected rather than guessed, because
for a communication aid saying nothing beats saying the wrong thing.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_SIZE, FEATURE_VERSION, extract_features
from .presets import classify_preset


@dataclass
class Prediction:
    label: str | None
    confidence: float
    source: str  # "custom", "preset" or "none"
    distance: float | None = None


class LibraryError(RuntimeError):
    pass


@dataclass
class GestureLibrary:
    """Recorded samples, stored as plain JSON so they're easy to back up or share."""

    samples: dict[str, list[list[float]]] = field(default_factory=dict)

    def add(self, label: str, features: np.ndarray) -> None:
        label = label.strip()
        if not label:
            raise LibraryError("gesture label can't be empty")
        vec = np.asarray(features, dtype=float).ravel()
        if vec.size != FEATURE_SIZE:
            raise LibraryError(f"expected {FEATURE_SIZE} features, got {vec.size}")
        self.samples.setdefault(label, []).append([round(float(v), 5) for v in vec])

    def add_landmarks(self, label: str, landmarks, handedness: str | None = None) -> None:
        self.add(label, extract_features(landmarks, handedness))

    def remove(self, label: str) -> bool:
        return self.samples.pop(label, None) is not None

    @property
    def labels(self) -> list[str]:
        return sorted(self.samples)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.samples.items())}

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        rows, labels = [], []
        for label, vecs in self.samples.items():
            rows.extend(vecs)
            labels.extend([label] * len(vecs))
        if not rows:
            return np.empty((0, FEATURE_SIZE)), []
        return np.asarray(rows, dtype=float), labels

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        return {"format": "cognisign-gestures", "feature_version": FEATURE_VERSION,
                "samples": self.samples}

    @classmethod
    def from_json(cls, data: dict) -> GestureLibrary:
        if data.get("format") != "cognisign-gestures":
            raise LibraryError("not a CogniSign gesture file")
        if data.get("feature_version") != FEATURE_VERSION:
            raise LibraryError(
                f"gesture file uses feature version {data.get('feature_version')}, "
                f"this build needs {FEATURE_VERSION}; please re-record your gestures"
            )
        lib = cls()
        for label, vecs in data.get("samples", {}).items():
            for v in vecs:
                lib.add(label, np.asarray(v))
        return lib

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename so a crash never leaves a half-written library
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".gestures-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> GestureLibrary:
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))


class KNNClassifier:
    """Distance-gated k-nearest-neighbour vote over a :class:`GestureLibrary`."""

    def __init__(self, library: GestureLibrary, k: int = 5, max_distance: float = 1.2):
        self.k = k
        self.max_distance = max_distance
        self.refit(library)

    def refit(self, library: GestureLibrary) -> None:
        self._X, self._y = library.matrix()

    def predict(self, features: np.ndarray) -> Prediction:
        if not self._y:
            return Prediction(None, 0.0, "none")
        x = np.asarray(features, dtype=float).ravel()
        dists = np.linalg.norm(self._X - x, axis=1)
        k = min(self.k, len(self._y))
        nearest = np.argpartition(dists, k - 1)[:k]
        nearest = nearest[np.argsort(dists[nearest])]
        best = float(dists[nearest[0]])
        if best > self.max_distance:
            return Prediction(None, 0.0, "none", best)
        votes = Counter(self._y[i] for i in nearest if dists[i] <= self.max_distance)
        label, n = votes.most_common(1)[0]
        closeness = 1.0 - best / self.max_distance
        confidence = (n / k) * (0.5 + 0.5 * closeness)
        return Prediction(label, round(confidence, 3), "custom", best)


class Recogniser:
    """Your recordings first, built-in presets as the fallback."""

    def __init__(self, library: GestureLibrary | None = None, *, k: int = 5,
                 max_distance: float = 1.2, min_confidence: float = 0.6,
                 use_presets: bool = True):
        self.library = library or GestureLibrary()
        self.knn = KNNClassifier(self.library, k=k, max_distance=max_distance)
        self.min_confidence = min_confidence
        self.use_presets = use_presets

    def refresh(self) -> None:
        self.knn.refit(self.library)

    def predict(self, landmarks, handedness: str | None = None) -> Prediction:
        custom = self.knn.predict(extract_features(landmarks, handedness))
        if custom.label is not None and custom.confidence >= self.min_confidence:
            return custom
        if self.use_presets:
            preset = classify_preset(landmarks, handedness)
            if preset is not None:
                return Prediction(preset, 1.0, "preset")
        return Prediction(None, custom.confidence, "none", custom.distance)
