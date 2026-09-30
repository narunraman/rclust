"""Queue history: how long other jobs waited on each cluster.

`rclust learn` pulls Slurm's accounting records (`sacct -a`) from each cluster and keeps, per job, only
its shape (GPUs, CPUs, memory, time limit, partition) and timing (when it became eligible to run, when
it started). No user or account names are requested or stored.

A job's wait is Start - Eligible, not Start - Submit, so time spent held or waiting on a dependency
does not count. Jobs that are still pending, or were cancelled before starting, are kept as censored
observations ("waited at least this long"): dropping them would make waits look shorter than they are.
Summaries use the Kaplan-Meier estimator for the same reason.
"""

from __future__ import annotations

import base64
import gzip
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .cluster import JobSpec

logger = logging.getLogger(__name__)

SACCT_FIELDS = ["JobIDRaw", "Eligible", "Start", "End", "State", "Partition", "ReqTRES", "Timelimit", "Elapsed"]

# sacct can return 100k+ rows a day on a large cluster, so compress on the far side
SACCT_CMD = (
    "set -o pipefail; date +%s; date +%Y-%m-%dT%H:%M:%S; "
    "sacct -a -X -n -P -S {start} -E {end} -o " + ",".join(SACCT_FIELDS) + " | gzip -c | base64 -w0"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    cluster     TEXT NOT NULL,
    jobid       TEXT NOT NULL,
    eligible    INTEGER NOT NULL,   -- unix time the job became eligible to start
    wait_s      INTEGER NOT NULL,   -- seconds from eligible to start (or to now/cancel, if censored)
    started     INTEGER NOT NULL,   -- 1 if it started; 0 if the wait is censored
    state       TEXT,
    partition   TEXT,
    cpus        INTEGER,
    mem_mb      INTEGER,
    nodes       INTEGER,
    gpus        INTEGER NOT NULL DEFAULT 0,
    gpu_type    TEXT,
    timelimit_s INTEGER,
    elapsed_s   INTEGER,
    PRIMARY KEY (cluster, jobid)
);
CREATE TABLE IF NOT EXISTS pulls (
    cluster TEXT PRIMARY KEY,
    until   INTEGER NOT NULL        -- unix time of the last successful pull
);
"""


def history_path() -> Path:
    """~/.local/share/rclust/history.sqlite, honouring $XDG_DATA_HOME."""
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "rclust" / "history.sqlite"


# --- parsing sacct output -------------------------------------------------------------------------

def parse_duration(text: str) -> Optional[int]:
    """Slurm durations: [D-]HH:MM:SS, MM:SS, or MM (with optional .fraction). None if unlimited/unknown."""
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "PARTITION_LIMIT", "INVALID", "UNKNOWN", "NONE"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, *parts
    else:
        h, m, s = 0, parts[0], 0
    return int(days * 86400 + h * 3600 + m * 60 + s)


def parse_mem_mb(text: str) -> Optional[int]:
    m = re.fullmatch(r"([\d.]+)([KMGTP]?)", text or "")
    if not m:
        return None
    scale = {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}[m.group(2)]
    return int(float(m.group(1)) * scale)


def parse_tres(text: str) -> dict:
    """ReqTRES, e.g. 'billing=8,cpu=8,gres/gpu:h100=2,gres/gpu=2,mem=64G,node=1'."""
    out = {"cpus": None, "mem_mb": None, "nodes": None, "gpus": 0, "gpu_type": None}
    for item in (text or "").split(","):
        key, _, val = item.partition("=")
        if key == "cpu":
            out["cpus"] = int(val)
        elif key == "mem":
            out["mem_mb"] = parse_mem_mb(val)
        elif key == "node":
            out["nodes"] = int(val)
        elif key == "gres/gpu":
            out["gpus"] = int(val)
        elif key.startswith("gres/gpu:"):
            out["gpu_type"] = key.split(":", 1)[1]
            out["gpus"] = out["gpus"] or int(val)
    return out


def _ts(text: str, utc_offset: int) -> Optional[int]:
    """Cluster-local 'YYYY-MM-DDTHH:MM:SS' -> unix time. None for Unknown/None."""
    try:
        local = datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return int((local - datetime(1970, 1, 1)).total_seconds()) - utc_offset


def parse_sacct(text: str, now: int, utc_offset: int) -> List[dict]:
    """Rows of `sacct -P -o SACCT_FIELDS` -> job records. `now` and the offset come from the cluster."""
    jobs = []
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != len(SACCT_FIELDS):
            continue
        row = dict(zip(SACCT_FIELDS, cols))
        eligible = _ts(row["Eligible"], utc_offset)
        if eligible is None:  # held, or waiting on a dependency: not queueing yet
            continue
        start = _ts(row["Start"], utc_offset)
        states = row["State"].split()
        if not states:
            continue
        state = states[0]  # "CANCELLED by 123" -> "CANCELLED"
        if start is not None and start >= eligible:
            wait, started = start - eligible, 1
        elif state == "PENDING":
            wait, started = now - eligible, 0
        else:  # cancelled (or failed) before starting: censored at its end time
            end = _ts(row["End"], utc_offset)
            if end is None or end < eligible:
                continue
            wait, started = end - eligible, 0
        jobs.append({
            "jobid": row["JobIDRaw"], "eligible": eligible, "wait_s": wait, "started": started,
            "state": state, "partition": row["Partition"],
            "timelimit_s": parse_duration(row["Timelimit"]), "elapsed_s": parse_duration(row["Elapsed"]),
            **parse_tres(row["ReqTRES"]),
        })
    return jobs


# --- storage -------------------------------------------------------------------------------------

class History:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else history_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.path.chmod(0o600)
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def add(self, cluster: str, jobs: Iterable[dict], until: int) -> int:
        cols = ["jobid", "eligible", "wait_s", "started", "state", "partition", "cpus", "mem_mb",
                "nodes", "gpus", "gpu_type", "timelimit_s", "elapsed_s"]
        rows = [(cluster, *[j[c] for c in cols]) for j in jobs]
        with self.db:
            # a job seen pending last time may have started since: newer rows replace older ones
            self.db.executemany(
                f"INSERT OR REPLACE INTO jobs (cluster, {', '.join(cols)}) "
                f"VALUES ({', '.join('?' * (len(cols) + 1))})", rows)
            self.db.execute("INSERT OR REPLACE INTO pulls VALUES (?, ?)", (cluster, until))
        return len(rows)

    def last_pull(self, cluster: str) -> Optional[int]:
        row = self.db.execute("SELECT until FROM pulls WHERE cluster = ?", (cluster,)).fetchone()
        return row[0] if row else None

    def clusters(self) -> List[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT cluster FROM jobs ORDER BY cluster")]

    def waits(self, cluster: str, spec: JobSpec, since: int, loose: bool = False) -> List[Tuple[int, int]]:
        """(wait_s, started) for jobs shaped like `spec` that became eligible after `since`."""
        where, args = ["cluster = ?", "eligible >= ?"], [cluster, since]
        gpus = (spec.gpus or 0) * (spec.nodes or 1) if spec.gpus_per_node else spec.gpus or 0
        where.append("gpus = ?")
        args.append(gpus)
        if spec.gpu_type:
            where.append("gpu_type = ?")
            args.append(spec.gpu_type)
        if not loose:
            if gpus == 0 and spec.cpus:
                where.append("cpus BETWEEN ? AND ?")
                cpus = spec.cpus * (spec.tasks or 1)
                args += [max(1, cpus // 2), cpus * 2]
            if spec.partition:
                where.append("partition = ?")
                args.append(spec.partition)
            limit = parse_duration(spec.time) if spec.time else None
            if limit:
                where.append("timelimit_s BETWEEN ? AND ?")
                args += [limit // 2, limit * 2]
        sql = f"SELECT wait_s, started FROM jobs WHERE {' AND '.join(where)}"
        return list(self.db.execute(sql, args))

    def first_eligible(self, cluster: str, since: int) -> Optional[int]:
        """When the oldest job on record for `cluster` after `since` became eligible."""
        row = self.db.execute("SELECT MIN(eligible) FROM jobs WHERE cluster = ? AND eligible >= ?",
                              (cluster, since)).fetchone()
        return row[0] if row else None


# --- fetching ------------------------------------------------------------------------------------

def fetch(ssh, since_days: float = 2, since_ts: Optional[int] = None,
          timeout: int = 600) -> Tuple[List[dict], int]:
    """Pull accounting records from one cluster over an open SSH connection.

    `since_ts` (a previous pull) takes precedence over `since_days`; an hour of overlap catches jobs
    that changed state around the last pull. Returns (jobs, cluster's current unix time).
    """
    if since_ts is not None:
        start = f"now-{max(1, int((datetime.now().timestamp() - since_ts) / 60) + 60)}minutes"
    else:
        start = f"now-{max(1, int(since_days * 24 * 60))}minutes"
    code, out, err = ssh.execute_command(SACCT_CMD.format(start=start, end="now"), timeout=timeout,
                                       use_login_shell=True)
    if code != 0:
        raise RuntimeError(err or f"sacct failed with exit code {code}")
    lines = out.split("\n", 2)
    if len(lines) < 3:
        raise RuntimeError("unexpected sacct output")
    now = int(lines[0])
    local_now = datetime.strptime(lines[1].strip(), "%Y-%m-%dT%H:%M:%S")
    utc_offset = round(((local_now - datetime(1970, 1, 1)).total_seconds() - now) / 900) * 900
    text = gzip.decompress(base64.b64decode(lines[2].strip())).decode("utf-8", "replace") if lines[2].strip() else ""
    return parse_sacct(text, now, utc_offset), now


# --- summaries -----------------------------------------------------------------------------------

def km_quantiles(obs: Sequence[Tuple[int, int]], qs: Sequence[float]) -> List[Optional[int]]:
    """Kaplan-Meier quantiles of wait time. obs: (wait_s, started). None if the curve never gets there
    (too many jobs still waiting to say)."""
    events = sorted(obs, key=lambda o: (o[0], -o[1]))  # at ties, count starts before censorings
    at_risk, surv = len(events), 1.0
    out: List[Optional[int]] = [None] * len(qs)
    i = 0
    while i < len(events):
        t = events[i][0]
        starts = censored = 0
        while i < len(events) and events[i][0] == t:
            starts += events[i][1]
            censored += 1 - events[i][1]
            i += 1
        if starts:
            surv *= 1 - starts / at_risk
            for k, q in enumerate(qs):
                if out[k] is None and 1 - surv >= q - 1e-12:
                    out[k] = t
        at_risk -= starts + censored
    return out


def restricted_mean(obs: Sequence[Tuple[int, int]], cap: int = 24 * 3600) -> Optional[float]:
    """Average wait in seconds, with each wait counted as at most `cap`, from the Kaplan-Meier curve
    (so still-waiting jobs count properly). This is what the history policy compares clusters on:
    unlike the median it reflects the long tail, and unlike a plain mean it is defined even when
    some jobs haven't started yet."""
    if not obs:
        return None
    events = sorted(obs, key=lambda o: (o[0], -o[1]))
    at_risk, surv, t_prev, area = len(events), 1.0, 0.0, 0.0
    i = 0
    while i < len(events) and events[i][0] < cap:
        t = events[i][0]
        area += surv * (t - t_prev)
        t_prev = t
        starts = censored = 0
        while i < len(events) and events[i][0] == t:
            starts += events[i][1]
            censored += 1 - events[i][1]
            i += 1
        if starts:
            surv *= 1 - starts / at_risk
        at_risk -= starts + censored
    return area + surv * (cap - t_prev)


@dataclass
class Summary:
    cluster: str
    jobs: int
    waiting: int              # still pending or cancelled while pending (censored)
    median: Optional[int]
    p80: Optional[int]
    loose: bool               # matched on GPUs only, because too few jobs matched the full shape
    mean24: Optional[float] = None   # average wait, each capped at 24h (restricted_mean)
    span_s: Optional[int] = None     # how far back this cluster's history reaches (at most `days`)


def summarize(history: History, clusters: Sequence[str], spec: JobSpec, days: float = 7,
              min_jobs: int = 20) -> List[Summary]:
    now = datetime.now()
    since = int((now - timedelta(days=days)).timestamp())
    out = []
    for name in clusters:
        obs, loose = history.waits(name, spec, since), False
        if len(obs) < min_jobs:
            obs, loose = history.waits(name, spec, since, loose=True), True
        median, p80 = km_quantiles(obs, [0.5, 0.8]) if obs else (None, None)
        first = history.first_eligible(name, since)
        span = max(0, int(now.timestamp()) - first) if first is not None else None
        out.append(Summary(name, len(obs), sum(1 - s for _, s in obs), median, p80, loose,
                           restricted_mean(obs), span))
    return out


def fmt_span(seconds: Optional[float], days: float = 7) -> str:
    """How much history a summary covers: '7 days', '3.5 days', '19 hours'. The window asked for
    (`days`) is shown as is when the history reaches back that far (or is unknown)."""
    if seconds is None or seconds >= days * 86400 - 3600:
        return f"{days:g} day{'s' if days != 1 else ''}"
    if seconds >= 1.95 * 86400:
        return f"{round(seconds / 86400, 1):g} days"
    if seconds >= 3600:
        hours = round(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    minutes = max(1, round(seconds / 60))
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def fmt_wait(seconds: Optional[int]) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return "<1m"
    h, m = divmod(seconds // 60, 60)
    if h >= 48:
        return f"{h // 24}d{h % 24:02d}h"
    return f"{h}h{m:02d}" if h else f"{m}m"
