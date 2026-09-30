"""Every choice comes with a one-line reason; --explain says how much history it covers."""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from click.testing import CliRunner

from rclust.api import JobHandle, Scheduler
from rclust.cli import main
from rclust.cluster import ClusterMetrics, JobSpec
from rclust.dispatcher import BalancedPolicy, FastestStartPolicy, HistoryPolicy
from rclust.history import History, fmt_span, summarize


class C:
    def __init__(self, name):
        self.name = name


def _metrics(minutes, fairshare=0.5, load=0):
    return ClusterMetrics(load=load, fairshare=fairshare,
                          estimated_start_time=datetime.now() + timedelta(minutes=minutes))


def _history(path, waits, age_s=3600, gpu_type=None):
    """waits: cluster -> list of (wait_s, started) for 2-GPU, 12h jobs that became eligible age_s ago."""
    h = History(path)
    now = int(time.time())
    for cluster, obs in waits.items():
        h.add(cluster, [dict(jobid=f"{cluster}{i}", eligible=now - age_s - i, wait_s=w, started=s,
                             state="COMPLETED" if s else "PENDING", partition="gpu", cpus=8, mem_mb=1,
                             nodes=1, gpus=2, gpu_type=gpu_type, timelimit_s=12 * 3600, elapsed_s=1)
                        for i, (w, s) in enumerate(obs)], until=now)
    return h


SPEC = JobSpec(gpus=2, time="12:00:00")


def test_history_reason_can_start_now(tmp_path):
    h = _history(tmp_path / "h.sqlite", {"a": [(4 * 3600, 1)] * 30, "b": [(600, 1)] * 30})
    policy = HistoryPolicy(SPEC, h)
    a, b = C("a"), C("b")
    assert policy.select({a: _metrics(1), b: _metrics(240)}) is a
    assert policy.reason == "a can start it now"


def test_history_reason_names_the_span_and_median(tmp_path):
    # only ~3 days of history: say so rather than "past week"
    h = _history(tmp_path / "h.sqlite", {"a": [(4 * 3600, 1)] * 30, "b": [(60, 1)] * 30}, age_s=3 * 86400)
    policy = HistoryPolicy(SPEC, h)
    assert policy.select({C("a"): _metrics(60), C("b"): _metrics(240)}).name == "b"
    assert policy.reason == "jobs like this waited least on b over the past 3 days (median 1m)"


def test_history_reason_says_week_when_history_is_full(tmp_path):
    h = _history(tmp_path / "h.sqlite", {"a": [(4 * 3600, 1)] * 30, "b": [(600, 1)] * 30},
                 age_s=int(6.9 * 86400))
    policy = HistoryPolicy(SPEC, h)
    policy.select({C("a"): _metrics(60), C("b"): _metrics(240)})
    assert policy.reason == "jobs like this waited least on b over the past week (median 10m)"


def test_history_reason_without_history(tmp_path):
    policy = HistoryPolicy(SPEC, History(tmp_path / "empty.sqlite"))
    policy.select({C("a"): _metrics(90), C("b"): _metrics(30)})
    assert policy.reason.startswith("soonest estimated start (no queue history")


def test_other_policies_give_reasons():
    earliest = FastestStartPolicy()
    earliest.select({C("a"): _metrics(90), C("b"): _metrics(30)})
    assert earliest.reason == "soonest estimated start"
    balanced = BalancedPolicy()
    balanced.select({C("a"): _metrics(10, fairshare=0.9, load=3), C("b"): _metrics(10, fairshare=0.2)})
    assert balanced.reason == "highest fairshare (0.90) with few of your jobs running (3)"


def _api(tmp_path, policy="earliest"):
    path = tmp_path / "config.yaml"
    path.write_text(f"default_policy: {policy}\nclusters:\n  cluster-a: {{}}\n  cluster-c: {{}}\n")
    api = Scheduler(str(path), project_dir=tmp_path / "state")
    for cluster, minutes in zip(api.dispatcher.clusters, (5, 90)):
        cluster._ssh_provider = lambda _: Mock()
        cluster.get_metrics = Mock(return_value=_metrics(minutes))
    return api


def test_suggest_prints_policy_and_reason(tmp_path):
    api = _api(tmp_path)
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["suggest"])
    assert result.exit_code == 0, result.output
    assert "earliest: soonest estimated start" in result.output
    assert "cluster-a" in result.output


def test_submit_prints_policy_and_reason(tmp_path):
    api = _api(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    api.submit = Mock(return_value=JobHandle("42", api.dispatcher.clusters[0], JobSpec()))
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["submit", str(script)])
    assert result.exit_code == 0, result.output
    assert "Selected cluster-a" in result.output
    assert "earliest: soonest estimated start" in result.output


def test_fmt_span():
    assert fmt_span(None, 7) == "7 days"
    assert fmt_span(7 * 86400, 7) == "7 days"
    assert fmt_span(3.5 * 86400, 7) == "3.5 days"
    assert fmt_span(19 * 3600, 7) == "19 hours"
    assert fmt_span(1 * 3600, 7) == "1 hour"
    assert fmt_span(90, 7) == "2 minutes"


def test_summaries_report_their_span(tmp_path):
    h = _history(tmp_path / "h.sqlite", {"a": [(60, 1)] * 5}, age_s=2 * 86400)
    (s,) = summarize(h, ["a"], SPEC, days=7)
    assert abs(s.span_s - 2 * 86400 - 4) < 5
    (s,) = summarize(h, ["a"], SPEC, days=1)  # older jobs fall outside the window
    assert s.span_s is None and s.jobs == 0


def test_explain_title_says_actual_span_and_gpu_type(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    from rclust.history import history_path

    _history(history_path(), {"cluster-a": [(60, 1)] * 30}, age_s=19 * 3600, gpu_type="h100").close()
    api = _api(tmp_path)
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["suggest", "--gpus", "h100:2", "--time", "12:00:00",
                                           "--explain"])
    assert result.exit_code == 0, result.output
    assert "Jobs like yours (2 h100 GPUs, ~12:00:00), last 19 hours" in result.output
