"""Read-only checks against real clusters from your rclust config (opt-in; see clusters.py).

Run with, e.g.:

    RCLUST_TEST_CLUSTERS=all uv run pytest tests/integration -v
    RCLUST_TEST_CLUSTERS=cluster-a,cluster-b uv run pytest tests/integration -v

Nothing here submits a job: probes use `sbatch --test-only`.
"""

import re
import time
from datetime import datetime

import pytest
import yaml

pytestmark = pytest.mark.cluster

EPOCH = datetime(1970, 1, 1)


def _run(cluster, command, login=False, timeout=60):
    code, out, err = cluster.ssh.execute_command(command, timeout=timeout, use_login_shell=login)
    assert code == 0, f"{cluster.name}: `{command}` exited {code}: {err or out}"
    return out


def _config_gpu_types(cluster):
    """GPU types the config lists for this cluster (untyped 'generic' GPUs can't be asked for)."""
    res = cluster.resources or {}
    types = list(res.get("gpus") or []) + [g.get("type") for g in res.get("gpu_types") or []]
    return [t for t in dict.fromkeys(types) if t and t != "generic"]


def test_connection_is_open(cluster):
    assert cluster.ssh.is_open()


def test_sinfo(cluster):
    out = _run(cluster, "sinfo --noheader -o '%P %a'")
    assert out.strip(), f"{cluster.name}: sinfo listed no partitions"


def test_squeue(cluster):
    _run(cluster, "squeue -u $USER --noheader -o '%i %T'")
    assert isinstance(cluster._get_load(), int)  # rclust's own call


def test_sshare(cluster):
    code, out, err = cluster.ssh.execute_command("sshare -u $USER -n -o LevelFS")
    if code == 127 or "command not found" in err:
        pytest.skip(f"{cluster.name} has no sshare (rclust then uses fairshare 0.5)")
    assert code == 0, f"{cluster.name}: sshare exited {code}: {err or out}"
    fairshare = cluster._get_fairshare()  # rclust's own call
    assert fairshare is not None and fairshare >= 0


def test_discover(cluster, run_rclust):
    result = run_rclust("discover", cluster.name)
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.stdout)
    entry = data["clusters"][cluster.name] or {}
    gpus = (entry.get("resources") or {}).get("gpus") or []
    assert all(isinstance(g, str) and g for g in gpus)


def test_sacct_history(cluster):
    from rclust import history as hist

    assert "xdg-data" in str(hist.history_path())  # a temporary DB, not the user's
    jobs, now = hist.fetch(cluster.ssh, since_days=1 / 24, timeout=300)  # the last hour
    assert abs(now - time.time()) < 300, f"{cluster.name}'s clock is {now - time.time():+.0f}s off"
    for job in jobs:
        assert job["wait_s"] >= 0 and job["jobid"]
        if job["started"]:  # a time-zone error would put starts in the future
            assert job["eligible"] + job["wait_s"] <= now + 300, job
    with hist.History() as history:
        assert history.add(cluster.name, jobs, now) == len(jobs)
        assert history.last_pull(cluster.name) == now


def test_start_estimate_and_time_zone(cluster):
    """`sbatch --test-only` for 1 CPU for 1 minute, and rclust's conversion to local time.

    The expected start is computed independently from the probe's own output: sbatch's estimate
    (cluster-local time) and the cluster clock read in the same shell (`date +%s` and local time).
    """
    from rclust.cluster import JobSpec

    spec = JobSpec(cpus=1, time="0:01:00")
    start = cluster._get_estimated_start_time(spec)
    assert start is not None, f"{cluster.name}: probe failed: {cluster.last_error}"

    cmd, _, out, err = next(e for e in reversed(cluster.ssh.log) if "sbatch --test-only" in e[0])
    account = cluster.cpu_account or cluster.account
    if account:
        assert f"--account={account}" in cmd
    text = out + "\n" + err
    clock = re.search(r"RCLUST_NOW (\d+) (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", text)
    estimate = re.search(r"start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", text)
    assert clock, f"{cluster.name}: no clock reading in the probe output"
    assert estimate, f"{cluster.name}: sbatch gave no dated estimate: {text.strip()[:300]}"

    cluster_epoch = int(clock.group(1))
    utc_offset = (datetime.fromisoformat(clock.group(2)) - EPOCH).total_seconds() - cluster_epoch
    expected = (datetime.fromisoformat(estimate.group(1)) - EPOCH).total_seconds() - utc_offset
    got = start.timestamp()  # rclust returns naive local time

    assert abs(cluster_epoch - time.time()) < 300, f"{cluster.name}'s clock is far from ours"
    assert abs(got - time.time()) <= 86400, f"{cluster.name}: estimate {start} is not within a day"
    assert abs(got - expected) < 600, (
        f"{cluster.name}: rclust says {start}, the cluster's own clock says "
        f"{datetime.fromtimestamp(expected)} ({(got - expected) / 3600:+.2f} h; a time-zone error?)")


@pytest.mark.parametrize("form", ["--gpu-type", "--gpus"])
def test_gpu_type_probe(cluster, rclust_config, run_rclust, form):
    types = _config_gpu_types(cluster)
    if not types:
        pytest.skip(f"the config lists no GPU types for {cluster.name}")
    gpu = types[0]
    flags = ["--gpu-type", gpu] if form == "--gpu-type" else ["--gpus", f"{gpu}:1"]
    others = [a for c in rclust_config.clusters if c.name != cluster.name for a in ("-x", c.name)]
    result = run_rclust("suggest", *flags, "--time", "0:01:00", "--policy", "earliest", *others)
    assert result.exit_code == 0, result.output
    assert cluster.name in result.stdout
    assert "skipped" not in result.output
