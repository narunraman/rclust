"""Cluster selection: filters and the earliest/balanced policies, with SSH mocked out."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rclust import JobSpec, Scheduler
from rclust.cluster import ClusterMetrics
from rclust.dispatcher import QueueTimePolicy

CONFIG = """
clusters:
  fast:
    host: fast.example.org
    tags: [gpu]
  fair:
    host: fair.example.org
    tags: [cpu]
default_policy: balanced
"""


@pytest.fixture
def scheduler(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return Scheduler(str(path), project_dir=tmp_path / "state")


def fake_ssh(replies):
    """subprocess.run stand-in: replies[host][command word] -> stdout (return code 0)."""
    def run(cmd, **kwargs):
        line = " ".join(cmd)
        for host, by_cmd in replies.items():
            if host in line:
                for word, out in by_cmd.items():
                    if word in line:
                        return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=0, stdout="0", stderr="")
    return run


REPLIES = {
    "fast.example.org": {"sshare": "0.1", "squeue": "10",
                         "--test-only": "sbatch: Job 1 to start at 2030-01-01T10:00:00"},
    "fair.example.org": {"sshare": "0.9", "squeue": "50",
                         "--test-only": "sbatch: Job 2 to start at 2030-01-01T12:00:00"},
}


@patch("rclust.ssh.subprocess.run")
def test_balanced_prefers_fairshare_over_an_earlier_start(run, scheduler):
    run.side_effect = fake_ssh(REPLIES)
    assert scheduler.propose_cluster(JobSpec(cpus=1), policy="balanced").name == "fair"


@patch("rclust.ssh.subprocess.run")
def test_earliest_and_its_old_names_pick_the_soonest_start(run, scheduler):
    run.side_effect = fake_ssh(REPLIES)
    for policy in ("earliest", "rush", "queue-time"):
        assert scheduler.propose_cluster(JobSpec(cpus=1), policy=policy).name == "fast", policy


@patch("rclust.ssh.subprocess.run")
def test_tags_and_exclusions_filter_before_probing(run, scheduler):
    run.side_effect = fake_ssh(REPLIES)
    assert scheduler.propose_cluster(JobSpec(cpus=1), tags=["cpu"]).name == "fair"
    assert not [c for c in run.call_args_list if "fast.example.org" in str(c) and "-G" not in str(c)]
    cluster, _ = scheduler.select_cluster(JobSpec(cpus=1), exclude=["fair"])
    assert cluster.name == "fast"
    assert scheduler.select_cluster(JobSpec(cpus=1), exclude=["fast", "fair"]) == (None, None)
    assert "exclusions" in scheduler.dispatcher.last_problem


@patch("rclust.ssh.subprocess.run")
def test_missing_sshare_does_not_rule_a_cluster_out(run, scheduler):
    def no_sshare(cmd, **kwargs):
        if "sshare" in " ".join(cmd):
            return SimpleNamespace(returncode=127, stdout="", stderr="sshare: command not found")
        return fake_ssh(REPLIES)(cmd)
    run.side_effect = no_sshare
    cluster, metrics = scheduler.select_cluster(JobSpec(cpus=1), policy="earliest")
    assert cluster.name == "fast" and metrics.fairshare == 0.5


class Named:
    def __init__(self, name):
        self.name = name


def test_queue_time_policy_ignores_fairshare_and_load():
    now = datetime.now()
    a, b = Named("a"), Named("b")
    candidates = {a: ClusterMetrics(9999, 0.001, now + timedelta(minutes=1)),
                  b: ClusterMetrics(1, 0.999, now + timedelta(hours=1))}
    assert QueueTimePolicy().select(candidates) is a
    assert QueueTimePolicy().select({}) is None
