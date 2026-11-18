import pytest
import yaml

from cognisign.config import (
    ConfigError,
    default_config_text,
    load_config,
    parse_config,
    write_default_config,
)
from cognisign.presets import PRESET_GESTURES


def test_default_config_loads_and_maps_every_preset(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert set(PRESET_GESTURES) <= set(cfg.gestures)
    assert cfg.action_for("thumbs_up").value == "Yes"
    assert cfg.action_for(None) is None


def test_init_writes_once(tmp_path):
    p = tmp_path / "c.yaml"
    write_default_config(p)
    p.write_text("gestures: {}\n")
    write_default_config(p)  # no overwrite
    assert p.read_text() == "gestures: {}\n"
    write_default_config(p, overwrite=True)
    assert p.read_text() == default_config_text()


@pytest.mark.parametrize("data, msg", [
    ({"nope": {}}, "unknown section"),
    ({"camera": {"zoom": 2}}, "unknown setting"),
    ({"camera": {"mirror": "yes"}}, "true or false"),
    ({"recognition": {"hold_ms": "long"}}, "number"),
    ({"recognition": {"hold_ms": -5}}, "negative"),
    ({"tracking": {"num_hands": 3}}, "num_hands"),
    ({"gestures": {"fist": {"value": "x"}}}, "needs an 'action'"),
    ({"gestures": {"fist": {"action": "shell", "value": "x"}}}, "unknown action"),
])
def test_bad_config_gives_helpful_errors(data, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(data)


def test_invalid_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def test_partial_config_keeps_defaults():
    cfg = parse_config(yaml.safe_load("recognition:\n  hold_ms: 900\n"))
    assert cfg.recognition.hold_ms == 900
    assert cfg.recognition.cooldown_ms == 700
    assert cfg.camera.mirror is True
