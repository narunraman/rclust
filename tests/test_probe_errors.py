"""Failed probes: readable reasons, and the user is always told which clusters were skipped."""

from datetime import datetime
from unittest.mock import Mock, patch

from click.testing import CliRunner

from rclust.api import Scheduler
from rclust.cli import main
from rclust.cluster import ClusterMetrics, JobSpec, SlurmCluster, error_lines, summarize_error
from rclust.config import ClusterConfig

BANNER = (
    "\x1b[1;31m--------------------------------------------------------------\x1b[0m\n"
    "\x1b[1;31m==============================================================\x1b[0m\n"
    "\x1b[33msbatch: error: GPU type required (e.g. --gpus=h100:1)\x1b[0m\n"
    "\x1b[1;31m--------------------------------------------------------------\x1b[0m\n"
    "sbatch: error: Batch job submission failed: Unspecified error\n"
)


def test_error_lines_strip_colour_and_decoration():
    assert error_lines(BANNER) == ["sbatch: error: GPU type required (e.g. --gpus=h100:1)",
                                   "sbatch: error: Batch job submission failed: Unspecified error"]
    assert summarize_error(BANNER) == "sbatch: error: GPU type required (e.g. --gpus=h100:1)"
    generic = "sbatch: error: Batch job submission failed: Invalid account"
    assert summarize_error(generic + "\n") == generic


def test_summarize_error_falls_back_to_first_meaningful_line():
    assert summarize_error("\n=====\nPermission denied\n") == "Permission denied"
    assert summarize_error("") == ""


def test_probe_records_readable_reason():
    ssh = Mock()
    ssh.execute_command.return_value = (1, "RCLUST_NOW 1 2026-01-01T00:00:00", BANNER)
    cluster = SlurmCluster(ClusterConfig("cluster-b", "cluster-b"), ssh_provider=lambda _: ssh)
    assert cluster.get_metrics(JobSpec(gpus=1, time="1:00:00")) is None
    assert "\x1b" not in cluster.last_error
    assert cluster.last_error.startswith("sbatch: error: GPU type required (e.g. --gpus=h100:1)")
    assert "---" not in cluster.last_error


def test_cli_always_says_which_clusters_were_skipped(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("clusters:\n  cluster-a: {}\n  cluster-b: {}\n")
    api = Scheduler(str(config), project_dir=tmp_path / "state")
    cluster_a, cluster_b = api.dispatcher.clusters
    good, bad = Mock(), Mock()
    cluster_a._ssh_provider = lambda _: good
    cluster_b._ssh_provider = lambda _: bad
    cluster_a.get_metrics = Mock(return_value=ClusterMetrics(0, 0.5, datetime.now()))
    bad.execute_command.return_value = (1, "", BANNER)
    with patch("rclust.cli.get_scheduler", return_value=api), \
            patch("rclust.dispatcher.HistoryPolicy.scores", return_value={}):
        result = CliRunner().invoke(main, ["suggest", "--gpus", "1"])
    assert result.exit_code == 0, result.output
    assert "skipped cluster-b: sbatch: error: GPU type required (e.g. --gpus=h100:1)" in result.output
    assert "\x1b[33m" not in result.output
    assert "cluster-a" in result.output
