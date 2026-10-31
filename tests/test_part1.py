"""Tests for Part 1 – repo skeleton and src.load_config / AttrDict."""

import os
from pathlib import Path

import pytest
import yaml

from src import AttrDict, load_config


# ---------------------------------------------------------------------------
# AttrDict
# ---------------------------------------------------------------------------

class TestAttrDict:
    def test_attribute_access(self):
        d = AttrDict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_nested_conversion(self):
        d = AttrDict({"outer": {"inner": 42}})
        assert d.outer.inner == 42
        assert isinstance(d.outer, AttrDict)

    def test_missing_key_raises_attribute_error(self):
        d = AttrDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = d.nonexistent

    def test_setattr(self):
        d = AttrDict()
        d.x = 10
        assert d["x"] == 10

    def test_delattr(self):
        d = AttrDict({"a": 1})
        del d.a
        assert "a" not in d

    def test_delattr_missing_raises(self):
        d = AttrDict()
        with pytest.raises(AttributeError):
            del d.missing


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_default_config(self):
        cfg = load_config()
        assert cfg["can"]["channel"] == "cavshield"

    def test_attribute_style(self):
        cfg = load_config()
        assert cfg.can.channel == "cavshield"
        assert cfg.simulation.world_hz == 200

    def test_paths_resolved_to_absolute(self):
        cfg = load_config()
        for key, val in cfg.paths.items():
            assert os.path.isabs(val), f"paths.{key} should be absolute: {val}"

    def test_paths_directories_exist(self):
        cfg = load_config()
        for key, val in cfg.paths.items():
            assert os.path.isdir(val), f"paths.{key} directory should exist: {val}"

    def test_overrides_deep_merge(self):
        cfg = load_config(overrides={"can": {"channel": "overridden"}})
        assert cfg.can.channel == "overridden"
        # Other keys in 'can' should still be present
        assert cfg.can.interface == "virtual"

    def test_all_top_level_sections_present(self):
        cfg = load_config()
        expected = [
            "simulation", "can", "v2x", "sensors", "features", "rules",
            "physics", "risk", "failsafe", "ai", "training", "attacks",
            "dashboard", "paths",
        ]
        for section in expected:
            assert section in cfg, f"Missing config section: {section}"

    def test_risk_flags(self):
        cfg = load_config()
        assert cfg.risk.use_ai is True
        assert cfg.risk.use_can_rules is True
        assert cfg.risk.use_v2x is True
        assert cfg.risk.use_physics is True


# ---------------------------------------------------------------------------
# Config YAML structure
# ---------------------------------------------------------------------------

class TestConfigYAML:
    @pytest.fixture()
    def raw_yaml(self):
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_simulation_defaults(self, raw_yaml):
        sim = raw_yaml["simulation"]
        assert sim["world_hz"] == 200
        assert sim["cruise_kmh"] == 72
        assert sim["degraded_kmh"] == 50
        assert sim["safe_kmh"] == 30
        assert sim["driver_brake_peak_pct"] == 22
        assert sim["brake_gain"] == 0.1
        assert sim["command_timeout_s"] == 0.06
        assert sim["protection_enabled"] is True

    def test_attacks_flood_ids(self, raw_yaml):
        flood = raw_yaml["attacks"]["flood"]
        assert flood["rate_hz"] == 700
        assert 0x000 in flood["ids"]
        assert 0x0A0 in flood["ids"]

    def test_dashboard_defaults(self, raw_yaml):
        dash = raw_yaml["dashboard"]
        assert dash["host"] == "127.0.0.1"
        assert dash["port"] == 8000

