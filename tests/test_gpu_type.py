"""--gpu-type and --gpus TYPE:N on the command line."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from rclust.api import JobHandle, Scheduler
from rclust.cli import main
from rclust.cluster import ClusterMetrics, JobSpec

CONFIG = """
clusters:
  cluster-a:
    gpu_types:
      - type: h100
        slurm_spec: nvidia_h100_80gb_hbm3
"""


@pytest.fixture
def api(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return Scheduler(str(path), project_dir=tmp_path / "state")


def _submit_spec(api, tmp_path, *args):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --time=2:00:00\ntrue\n")
    cluster = api.dispatcher.clusters[0]
    cluster._ssh_provider = lambda _: Mock()
    api.select_cluster = Mock(return_value=(cluster, ClusterMetrics(0, 1, datetime.now())))
    api.submit = Mock(return_value=JobHandle("42", cluster, JobSpec()))
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["submit", str(script), *args])
    return result, (api.select_cluster.call_args.args[0] if api.select_cluster.called else None)


@pytest.mark.parametrize("args, expected", [
    (["--gpus", "h100:2"], ("h100", 2)),
    (["--gpus", "2", "--gpu-type", "h100"], ("h100", 2)),
    (["--gpu-type", "h100"], ("h100", 1)),
    (["--gpus", "h100:2", "--gpu-type", "h100"], ("h100", 2)),
    (["--gpus", "3"], (None, 3)),
])
def test_submit_accepts_gpu_type(api, tmp_path, args, expected):
    result, spec = _submit_spec(api, tmp_path, *args)
    assert result.exit_code == 0, result.output
    assert (spec.gpu_type, spec.gpus) == expected


@pytest.mark.parametrize("args", [["--gpus", "h100:2", "--gpu-type", "a100"], ["--gpus", "h100:0"],
                                  ["--gpus", "0", "--gpu-type", "h100"], ["--gpus", ":2"],
                                  ["--gpus", "two"], ["--gpus", "a:b:2"], ["--gpu-type", "h 100"]])
def test_bad_gpu_requests_are_usage_errors(api, tmp_path, args):
    result, _ = _submit_spec(api, tmp_path, *args)
    assert result.exit_code == 2, result.output


def test_suggest_probes_with_the_configured_slurm_spec(api):
    ssh = Mock()
    ssh.execute_command.return_value = (0, "", "sbatch: Job 1 to start at 2020-01-01T00:00:00")
    api.dispatcher.clusters[0]._ssh_provider = lambda _: ssh
    with patch("rclust.cli.get_scheduler", return_value=api), \
            patch("rclust.dispatcher.HistoryPolicy.scores", return_value={}):
        result = CliRunner().invoke(main, ["suggest", "--gpus", "h100:2"])
    assert result.exit_code == 0, result.output
    probe = ssh.execute_command.call_args_list[0].args[0]
    assert "--gpus=nvidia_h100_80gb_hbm3:2" in probe


def test_suggest_skips_clusters_without_the_requested_type(api):
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["suggest", "--gpu-type", "a100"])
    assert result.exit_code == 1
    assert "no cluster has the GPUs" in result.output
