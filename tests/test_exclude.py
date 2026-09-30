"""-x/--exclude with a name that is not a configured cluster is an error, not silently ignored."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from rclust.api import Scheduler
from rclust.cli import main
from rclust.cluster import ClusterMetrics


@pytest.fixture
def api(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("clusters:\n  cluster-a: {}\n  cluster-c: {}\n")
    api = Scheduler(str(path), project_dir=tmp_path / "state")
    for cluster in api.dispatcher.clusters:
        cluster._ssh_provider = lambda _: Mock()
        cluster.get_metrics = Mock(return_value=ClusterMetrics(0, 0.5, datetime.now()))
    return api


@pytest.mark.parametrize("command", [["suggest"], ["submit", "JOB"]])
def test_unknown_exclude_is_an_error(api, tmp_path, command):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    command = [str(script) if a == "JOB" else a for a in command]
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, [*command, "-x", "fri"])
    assert result.exit_code == 1
    assert "unknown cluster(s) in --exclude: fri" in result.output
    assert "Configured: cluster-a, cluster-c" in result.output
    for cluster in api.dispatcher.clusters:
        cluster.get_metrics.assert_not_called()


def test_known_exclude_still_works(api):
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["suggest", "-x", "cluster-a"])
    assert result.exit_code == 0, result.output
    assert "cluster-c" in result.output
    api.dispatcher.clusters[0].get_metrics.assert_not_called()
