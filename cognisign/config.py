"""Config loading and the folders CogniSign keeps its data in."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .actions import Action


class ConfigError(ValueError):
    pass


def home_dir() -> Path:
    """~/.cognisign, overridable with COGNISIGN_HOME (handy for tests and portable installs)."""
    return Path(os.environ.get("COGNISIGN_HOME", Path.home() / ".cognisign"))


def config_path() -> Path:
    return home_dir() / "config.yaml"


def gestures_path() -> Path:
    return home_dir() / "gestures.json"


def models_dir() -> Path:
    return home_dir() / "models"


def default_config_text() -> str:
    return resources.files("cognisign").joinpath("default_config.yaml").read_text("utf-8")


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True


@dataclass
class TrackingConfig:
    num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass
class RecognitionConfig:
    hold_ms: int = 600
    cooldown_ms: int = 700
    min_confidence: float = 0.6
    k: int = 5
    max_distance: float = 1.2
    use_presets: bool = True


@dataclass
class VoiceConfig:
    rate: int = 170
    voice: str | None = None


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    gestures: dict[str, Action] = field(default_factory=dict)

    def action_for(self, label: str | None) -> Action | None:
        return self.gestures.get(label) if label else None


def _section(cls, data: dict | None, name: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a mapping")
    known = cls.__dataclass_fields__
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"[{name}] unknown setting(s): {', '.join(sorted(unknown))}")
    try:
        obj = cls(**data)
    except TypeError as exc:  # pragma: no cover - guarded above
        raise ConfigError(f"[{name}] {exc}") from exc
    for key, f in known.items():
        default = f.default
        val = getattr(obj, key)
        if isinstance(default, bool) and not isinstance(val, bool):
            raise ConfigError(f"[{name}] {key} must be true or false")
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ConfigError(f"[{name}] {key} must be a number")
            if val < 0:
                raise ConfigError(f"[{name}] {key} can't be negative")
    return obj


def parse_config(data: dict | None) -> Config:
    data = data or {}
    allowed = {"camera", "tracking", "recognition", "voice", "gestures"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(sorted(unknown))}")

    gestures: dict[str, Action] = {}
    for label, spec in (data.get("gestures") or {}).items():
        if not isinstance(spec, dict) or "action" not in spec:
            raise ConfigError(f"gesture {label!r} needs an 'action'")
        try:
            gestures[str(label)] = Action(spec["action"], str(spec.get("value", "") or ""))
        except ValueError as exc:
            raise ConfigError(f"gesture {label!r}: {exc}") from exc

    cfg = Config(
        camera=_section(CameraConfig, data.get("camera"), "camera"),
        tracking=_section(TrackingConfig, data.get("tracking"), "tracking"),
        recognition=_section(RecognitionConfig, data.get("recognition"), "recognition"),
        voice=_section(VoiceConfig, data.get("voice"), "voice"),
        gestures=gestures,
    )
    if not 1 <= cfg.tracking.num_hands <= 2:
        raise ConfigError("[tracking] num_hands must be 1 or 2")
    if cfg.recognition.k < 1:
        raise ConfigError("[recognition] k must be at least 1")
    return cfg


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the user's config, falling back to the bundled defaults."""
    p = Path(path) if path else config_path()
    text = p.read_text("utf-8") if p.exists() else default_config_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML ({exc})") from exc
    return parse_config(data)


def write_default_config(path: str | os.PathLike | None = None, overwrite: bool = False) -> Path:
    p = Path(path) if path else config_path()
    if p.exists() and not overwrite:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(default_config_text(), "utf-8")
    return p
