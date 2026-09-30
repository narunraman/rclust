"""The example config ships with the package and `rclust config --example` prints it."""

import yaml
from click.testing import CliRunner

import pytest

from rclust.cli import main
from rclust.config import ConfigError, GlobalConfig


def test_config_example_prints_a_valid_config(tmp_path):
    result = CliRunner().invoke(main, ["config", "--example"])
    assert result.exit_code == 0, result.output
    assert "clusters:" in result.stdout
    path = tmp_path / "config.yaml"
    path.write_text(result.stdout)
    config = GlobalConfig.load(str(path))
    assert [c.name for c in config.clusters] == ["cluster-a", "cluster-b"]
    assert yaml.safe_load(result.stdout)["default_policy"] == "history"


def test_config_example_needs_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("RCLUST_CONFIG", raising=False)
    result = CliRunner().invoke(main, ["config", "--example"])
    assert result.exit_code == 0
    assert not (tmp_path / "xdg").exists()  # prints only; writes nothing


def test_missing_config_hint_points_at_the_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("RCLUST_CONFIG", raising=False)
    with pytest.raises(ConfigError) as info:
        GlobalConfig.load()
    assert "rclust config --example" in str(info.value)
    assert "config.yaml.example" not in str(info.value)
