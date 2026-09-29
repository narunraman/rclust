import base64
import gzip
from datetime import datetime

import pytest

from rclust import history as hist
from rclust.cluster import JobSpec

NOW = 1_790_000_000


def row(jobid, eligible, start, end, state, tres="cpu=4,mem=16G,node=1", limit="12:00:00", elapsed="01:00:00"):
    return "|".join([jobid, eligible, start, end, state, "gpu", tres, limit, elapsed])


def test_parse_duration():
    assert hist.parse_duration("1-02:03:04") == 86400 + 7384
    assert hist.parse_duration("12:00:00") == 43200
    assert hist.parse_duration("30:00") == 1800
    assert hist.parse_duration("45") == 2700
    assert hist.parse_duration("UNLIMITED") is None


def test_parse_tres():
    t = hist.parse_tres("billing=8,cpu=8,gres/gpu:h100=2,gres/gpu=2,mem=64G,node=1")
    assert t == {"cpus": 8, "mem_mb": 65536, "nodes": 1, "gpus": 2, "gpu_type": "h100"}
    assert hist.parse_tres("cpu=1,mem=4000M,node=1")["gpus"] == 0


def test_parse_sacct_waits_and_censoring():
    text = "\n".join([
        row("1", "2026-09-28T10:00:00", "2026-09-28T10:30:00", "2026-09-28T11:30:00", "COMPLETED"),  # waited 30m
        row("2", "2026-09-28T11:00:00", "Unknown", "Unknown", "PENDING"),                         # 1h so far
        row("3", "2026-09-28T09:00:00", "None", "2026-09-28T09:20:00", "CANCELLED by 42"),       # gave up at 20m
        row("4", "Unknown", "Unknown", "Unknown", "PENDING"),                                    # held: skipped
        "garbage line",
    ])
    now = hist._ts("2026-09-28T12:00:00", 0)
    jobs = {j["jobid"]: j for j in hist.parse_sacct(text, now, utc_offset=0)}
    assert set(jobs) == {"1", "2", "3"}
    assert (jobs["1"]["wait_s"], jobs["1"]["started"]) == (1800, 1)
    assert (jobs["2"]["wait_s"], jobs["2"]["started"]) == (3600, 0)
    assert (jobs["3"]["wait_s"], jobs["3"]["started"], jobs["3"]["state"]) == (1200, 0, "CANCELLED")
    assert jobs["1"]["timelimit_s"] == 43200 and jobs["1"]["cpus"] == 4


def test_km_quantiles_without_censoring_is_the_empirical_quantile():
    obs = [(t, 1) for t in (10, 20, 30, 40, 50)]
    assert hist.km_quantiles(obs, [0.2, 0.5, 0.8, 1.0]) == [10, 30, 40, 50]


def test_km_quantiles_censoring_pushes_waits_up():
    # three jobs started quickly, but two are still waiting after a long time:
    # ignoring the waiting ones would claim 80% start within 3 minutes
    obs = [(60, 1), (120, 1), (180, 1), (10000, 0), (10000, 0)]
    median, p80 = hist.km_quantiles(obs, [0.5, 0.8])
    assert median == 180
    assert p80 is None  # can't say: too many still waiting


def test_history_upsert_and_shape_matching(tmp_path):
    h = hist.History(tmp_path / "h.sqlite")
    base = dict(eligible=NOW, state="COMPLETED", partition="gpu", cpus=8, mem_mb=1024, nodes=1,
                gpu_type="h100", elapsed_s=60)
    h.add("cluster-a", [dict(base, jobid="1", wait_s=100, started=0, gpus=2, timelimit_s=12 * 3600)], until=NOW)
    # later pull: job 1 has started; its row is replaced, not duplicated
    h.add("cluster-a", [dict(base, jobid="1", wait_s=500, started=1, gpus=2, timelimit_s=12 * 3600),
                   dict(base, jobid="2", wait_s=50, started=1, gpus=2, timelimit_s=72 * 3600),
                   dict(base, jobid="3", wait_s=10, started=1, gpus=0, timelimit_s=12 * 3600)], until=NOW + 5)
    assert h.last_pull("cluster-a") == NOW + 5
    spec = JobSpec(gpus=2, time="12:00:00")
    assert h.waits("cluster-a", spec, since=0) == [(500, 1)]                  # 72h job is outside 6h-24h
    assert sorted(h.waits("cluster-a", spec, since=0, loose=True)) == [(50, 1), (500, 1)]  # GPUs only
    assert h.waits("cluster-a", JobSpec(gpus=2, gpu_type="a100"), since=0) == []


def test_summarize_falls_back_to_loose_matching(tmp_path):
    h = hist.History(tmp_path / "h.sqlite")
    jobs = [dict(jobid=str(i), eligible=int(datetime.now().timestamp()), wait_s=60 * i, started=1,
                 state="COMPLETED", partition="gpu", cpus=4, mem_mb=1, nodes=1, gpus=1, gpu_type=None,
                 timelimit_s=3600 * 48, elapsed_s=1) for i in range(1, 31)]
    h.add("cluster-b", jobs, until=0)
    [s] = hist.summarize(h, ["cluster-b"], JobSpec(gpus=1, time="1:00:00"), days=7)
    assert s.loose and s.jobs == 30 and s.median == 15 * 60 and s.waiting == 0
    [empty] = hist.summarize(h, ["cluster-a"], JobSpec(gpus=1), days=7)
    assert empty.jobs == 0 and empty.median is None


class FakeSSH:
    def __init__(self, sacct_text):
        self.sacct_text, self.cmds = sacct_text, []

    def execute_command(self, cmd, timeout=15, **kwargs):
        self.cmds.append(cmd)
        payload = base64.b64encode(gzip.compress(self.sacct_text.encode())).decode()
        # cluster clock: 12:00 local, 7 hours behind UTC
        now = hist._ts("2026-09-28T12:00:00", -7 * 3600)
        return 0, f"{now}\n2026-09-28T12:00:00\n{payload}", ""


def test_fetch_decodes_and_uses_cluster_clock():
    ssh = FakeSSH(row("9", "2026-09-28T11:00:00", "Unknown", "Unknown", "PENDING"))
    jobs, now = hist.fetch(ssh, since_days=1)
    assert "sacct -a -X -n -P -S now-1440minutes" in ssh.cmds[0]
    assert "User" not in ssh.cmds[0] and "Account" not in ssh.cmds[0]
    assert len(jobs) == 1 and jobs[0]["wait_s"] == 3600 and jobs[0]["started"] == 0


def test_fetch_raises_on_failure():
    class Broken:
        def execute_command(self, cmd, timeout=15, **kwargs):
            return 1, "", "sacct: error: slurmdbd unavailable"
    with pytest.raises(RuntimeError, match="slurmdbd"):
        hist.fetch(Broken())


def test_fmt_wait():
    assert hist.fmt_wait(None) == "—"
    assert hist.fmt_wait(30) == "<1m"
    assert hist.fmt_wait(25 * 60) == "25m"
    assert hist.fmt_wait(100 * 60) == "1h40"
    assert hist.fmt_wait(3 * 86400 + 3600) == "3d01h"
