"""Plumbing for the opt-in real-cluster tests: which clusters to test, and SSH that never logs in.

Nothing here names a real cluster. The clusters come from your own rclust config at run time,
found the way rclust finds it (`--rclust-config` for pytest in place of rclust's `--config`, then
$RCLUST_CONFIG, ./config.yaml, ~/.config/rclust/config.yaml):

    RCLUST_TEST_CLUSTERS=cluster-a,cluster-b   # these configured clusters
    RCLUST_TEST_CLUSTERS=all                   # every configured cluster
    RCLUST_TEST_SUBMIT=1                       # also submit one tiny job to each (spends allocation)

The tests only reuse connections you already opened with `rclust connect`: clusters may need a
password or two-factor login, which a test run cannot answer. Every ssh call here runs with
BatchMode=yes, and each remote command first checks (`ssh -O check`) that the shared connection
is still open, so a test never starts a new login.
"""

import os
import subprocess
from typing import List, Optional, Sequence

import pytest

ENV_CLUSTERS = "RCLUST_TEST_CLUSTERS"
ENV_SUBMIT = "RCLUST_TEST_SUBMIT"

SKIP_REASON = (f"real-cluster test; set {ENV_CLUSTERS}=all (or cluster names from your rclust "
               "config) to run it")


_STASH_KEY = pytest.StashKey()


class SelectionError(ValueError):
    """RCLUST_TEST_CLUSTERS names a cluster that is not in the config."""


class NotConnected(RuntimeError):
    """There is no open shared SSH connection to the cluster."""


def connect_hint(name: str) -> str:
    return f"no open SSH connection to {name}; run `rclust connect {name}` first"


def requested(environ=None) -> Optional[str]:
    """The raw RCLUST_TEST_CLUSTERS value, or None when the real-cluster tests are not wanted."""
    value = (os.environ if environ is None else environ).get(ENV_CLUSTERS, "")
    return value.strip() or None


def submit_enabled(environ=None) -> bool:
    """RCLUST_TEST_SUBMIT=1 (exactly) allows the one tiny real job per cluster."""
    return (os.environ if environ is None else environ).get(ENV_SUBMIT, "").strip() == "1"


def parse_selection(value: Optional[str], configured: Sequence[str]) -> List[str]:
    """Cluster names to test from a RCLUST_TEST_CLUSTERS value.

    "all" means every configured cluster; otherwise a comma-separated list of configured names
    (order kept, duplicates and blanks dropped). Unknown names raise SelectionError, listing
    the configured clusters. None or blank selects nothing.
    """
    if value is None or not value.strip():
        return []
    names = [n.strip() for n in value.split(",") if n.strip()]
    if [n.lower() for n in names] == ["all"]:
        return list(configured)
    unknown = [n for n in names if n not in configured]
    if unknown:
        raise SelectionError(
            f"{ENV_CLUSTERS} names cluster(s) not in your rclust config: {', '.join(unknown)}. "
            f"Configured: {', '.join(configured) or '(none)'} (or use {ENV_CLUSTERS}=all)")
    return list(dict.fromkeys(names))


def check_command(client) -> List[str]:
    """The `ssh -O check` command for an rclust SSHClient's shared connection."""
    return client._get_base_flags() + ["-o", "BatchMode=yes", "-O", "check",
                                       "-S", str(client.socket_path), client.destination]


def master_open(client) -> bool:
    """True if the client's shared (ControlMaster) connection is open. Never opens one."""
    if not client.socket_path.exists():
        return False
    try:
        result = subprocess.run(check_command(client), capture_output=True, timeout=15,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def selection(config):
    """(GlobalConfig or None, [cluster names]) for this pytest run, resolved once.

    Raises pytest.UsageError (a clean one-line error) if RCLUST_TEST_CLUSTERS is set but there is
    no usable config, or it names clusters that are not configured.
    """
    key = _STASH_KEY
    if key not in config.stash:
        value = requested()
        if value is None:
            config.stash[key] = (None, [])
        else:
            from rclust.config import ConfigError, GlobalConfig

            try:
                rclust_config = GlobalConfig.load(config.getoption("--rclust-config", default=None))
            except ConfigError as e:
                raise pytest.UsageError(f"{ENV_CLUSTERS} is set, but: {e}") from None
            try:
                names = parse_selection(value, [c.name for c in rclust_config.clusters])
            except SelectionError as e:
                raise pytest.UsageError(str(e)) from None
            config.stash[key] = (rclust_config, names)
    return config.stash[key]


class GuardedSSH:
    """An rclust SSHClient that refuses to log in: only an already-open connection is used.

    `connect()` succeeds only if the connection is open (it never prompts), and every command or
    upload first checks that it still is. Commands and their results are kept in `log`.
    """

    def __init__(self, client):
        self._client = client
        self.log = []  # (command, exit code, stdout, stderr)

    def __getattr__(self, name):
        return getattr(self._client, name)

    @property
    def name(self) -> str:
        return self._client.name or self._client.host

    def is_open(self) -> bool:
        return master_open(self._client)

    def _require_open(self):
        if not self.is_open():
            raise NotConnected(connect_hint(self.name))

    def connect(self, persist=None):
        self._require_open()

    def execute_command(self, cmd, timeout=15, use_login_shell=False):
        self._require_open()
        code, out, err = self._client.execute_command(cmd, timeout=timeout,
                                                      use_login_shell=use_login_shell)
        self.log.append((cmd, code, out, err))
        return code, out, err

    def rsync(self, local_path, remote_path):
        self._require_open()
        return self._client.rsync(local_path, remote_path)
