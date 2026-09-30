"""`rclust discover`: robust GRES parsing and plain, copy-pasteable YAML."""

from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from rclust.api import Scheduler
from rclust.cli import main
from rclust.cluster import SlurmCluster, parse_gres_gpu_types
from rclust.config import ClusterConfig


@pytest.mark.parametrize("gres, expected", [
    ("gpu:a5000:4(S:0-1)", {"a5000"}),
    ("gpu:nvidia_h100_80gb_hbm3:4", {"nvidia_h100_80gb_hbm3"}),
    ("gpu:nvidia_h100_80gb_hbm3_1g.10gb:2(S:0),gpu:nvidia_h100_80gb_hbm3_3g.40gb:1(S:1)",
     {"nvidia_h100_80gb_hbm3_1g.10gb", "nvidia_h100_80gb_hbm3_3g.40gb"}),
    ("gpu:a100:4(S:0,1),shard:a100:16", {"a100"}),     # comma inside the socket list
    ("gpu:4", {"generic"}),
    ("gpu:a100:no_consume:2", {"a100"}),
    ("gres/gpu:v100:2", {"v100"}),
    ("(null)", set()),
    ("", set()),
    ("mps:100,gpu:a5000:4(IDX:0-3)", {"a5000"}),
])
def test_parse_gres_gpu_types(gres, expected):
    assert parse_gres_gpu_types(gres) == expected


def test_discover_reads_all_partitions_untruncated():
    ssh = Mock()
    ssh.execute_command.return_value = (0, "gpu:a5000:4(S:0-1)   \n(null)   \ngpu:h100:8(S:0-3)   \n", "")
    cluster = SlurmCluster(ClusterConfig("a", "a"), ssh_provider=lambda _: ssh)
    assert cluster.discover_resources() == {"gpus": ["a5000", "h100"]}
    command = ssh.execute_command.call_args.args[0]
    assert "--all" in command and "%1000G" in command


def test_discover_prints_plain_yaml_on_stdout(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("clusters:\n  cluster-a: {}\n  cluster-b: {}\n")
    api = Scheduler(str(path), project_dir=tmp_path / "state")
    a, b = api.dispatcher.clusters
    for cluster in (a, b):
        cluster._ssh_provider = lambda _: Mock()
    a.discover_resources = Mock(return_value={"gpus": ["a5000", "nvidia_h100_80gb_hbm3_1g.10gb"]})
    b.discover_resources = Mock(return_value={})
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["discover"])
    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert not any(ch in stdout for ch in "│╭╮╰╯─")
    assert stdout.startswith("clusters:\n")
    data = yaml.safe_load(stdout)
    assert data["clusters"]["cluster-a"]["resources"]["gpus"] == ["a5000", "nvidia_h100_80gb_hbm3_1g.10gb"]
    assert "cluster-b" in data["clusters"]
    assert "# MIG slice" in stdout
