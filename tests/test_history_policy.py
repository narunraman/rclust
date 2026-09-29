import time
from datetime import datetime, timedelta

import pytest

from rclust.cluster import ClusterMetrics, JobSpec, SlurmCluster
from rclust.config import ClusterConfig
from rclust.dispatcher import Dispatcher, HistoryPolicy
from rclust.history import History, restricted_mean


def test_restricted_mean():
    assert restricted_mean([(0, 1), (0, 1)]) == 0
    assert restricted_mean([(3600, 1)] * 4) == 3600
    assert restricted_mean([(30 * 3600, 0)] * 3) == 24 * 3600          # still waiting past the cap
    # half started at 1h; half still waiting at 2h (so at least 2h, counted up to the 24h cap)
    assert restricted_mean([(3600, 1), (7200, 0)]) == pytest.approx(3600 * 0.5 + 24 * 3600 * 0.5)
    assert restricted_mean([]) is None


class C:
    def __init__(self, name):
        self.name = name


def _history(tmp_path, waits):
    """waits: cluster -> list of (wait_s, started) for 2-GPU, 12h jobs in the past day."""
    h = History(tmp_path / "h.sqlite")
    now = int(time.time())
    for cluster, obs in waits.items():
        h.add(cluster, [dict(jobid=f"{cluster}{i}", eligible=now - 3600 - i, wait_s=w, started=s,
                             state="COMPLETED" if s else "PENDING", partition="gpu", cpus=8, mem_mb=1,
                             nodes=1, gpus=2, gpu_type=None, timelimit_s=12 * 3600, elapsed_s=1)
                        for i, (w, s) in enumerate(obs)], until=now)
    return h


def _metrics(minutes_from_now):
    return ClusterMetrics(load=0, fairshare=0.5, estimated_start_time=datetime.now() + timedelta(minutes=minutes_from_now))


SPEC = JobSpec(gpus=2, time="12:00:00")


def test_history_picks_the_cluster_where_jobs_like_this_waited_least(tmp_path):
    h = _history(tmp_path, {"a": [(4 * 3600, 1)] * 30, "b": [(600, 1)] * 30})
    a, b = C("a"), C("b")
    # Slurm's estimates favour a, but b is where these jobs have actually started sooner
    assert HistoryPolicy(SPEC, h).select({a: _metrics(60), b: _metrics(240)}) is b


def test_a_cluster_that_can_start_it_now_wins(tmp_path):
    h = _history(tmp_path, {"a": [(4 * 3600, 1)] * 30, "b": [(600, 1)] * 30})
    a, b = C("a"), C("b")
    assert HistoryPolicy(SPEC, h).select({a: _metrics(1), b: _metrics(240)}) is a
    # two can start now: history breaks the tie
    assert HistoryPolicy(SPEC, h).select({a: _metrics(1), b: _metrics(2)}) is b


def test_without_history_it_falls_back_to_slurms_estimate(tmp_path):
    h = History(tmp_path / "empty.sqlite")
    a, b = C("a"), C("b")
    assert HistoryPolicy(SPEC, h).select({a: _metrics(90), b: _metrics(30)}) is b


def test_history_is_the_default_and_learned_is_an_alias():
    assert Dispatcher.ALIASES["learned"] == "history"
    from rclust.config import GlobalConfig
    assert GlobalConfig(clusters=[]).default_policy == "history"


class FakeSSH:
    def __init__(self, out):
        self.out, self.cmds = out, []

    def execute_command(self, cmd, timeout=15, use_login_shell=False):
        self.cmds.append(cmd)
        return 0, "", self.out


def test_test_only_estimates_are_converted_from_the_clusters_time_zone():
    now = int(time.time())
    here = datetime.fromtimestamp(now)
    there = here + timedelta(hours=3)                  # the cluster's clock is 3 hours ahead of ours
    start_there = there + timedelta(minutes=10)
    out = f"sbatch: Job 1 to start at {start_there:%Y-%m-%dT%H:%M:%S}\nRCLUST_NOW {now} {there:%Y-%m-%dT%H:%M:%S}"
    ssh = FakeSSH(out)
    cluster = SlurmCluster(ClusterConfig(name="x", host="x", user="u"), ssh_provider=lambda name: ssh)
    est = cluster._get_estimated_start_time(JobSpec(gpus=1, time="1:00:00"))
    assert abs((est - (here + timedelta(minutes=10))).total_seconds()) < 2
    assert "rc=$?" in ssh.cmds[0] and "exit $rc" in ssh.cmds[0]
