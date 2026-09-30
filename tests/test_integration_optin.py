"""The opt-in plumbing of tests/integration, checked offline (no real cluster, no real ssh)."""

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from integration.clusters import (GuardedSSH, NotConnected, SelectionError, master_open,
                                  parse_selection, requested, submit_enabled)

REPO = Path(__file__).resolve().parents[1]
CONFIGURED = ["cluster-a", "cluster-b", "cluster-c"]


# --- selection ------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, []),
    ("", []),
    ("  ", []),
    ("cluster-b", ["cluster-b"]),
    ("cluster-c, cluster-a", ["cluster-c", "cluster-a"]),
    ("cluster-a,,cluster-a,", ["cluster-a"]),
    ("all", CONFIGURED),
    (" ALL ", CONFIGURED),
])
def test_parse_selection(value, expected):
    assert parse_selection(value, CONFIGURED) == expected


def test_unknown_cluster_is_an_error_listing_configured_ones():
    with pytest.raises(SelectionError) as e:
        parse_selection("cluster-a,cluster-z", CONFIGURED)
    message = str(e.value)
    assert "cluster-z" in message and "cluster-a, cluster-b, cluster-c" in message


def test_opt_in_variables():
    assert requested({}) is None and requested({"RCLUST_TEST_CLUSTERS": " "}) is None
    assert requested({"RCLUST_TEST_CLUSTERS": "all"}) == "all"
    assert submit_enabled({"RCLUST_TEST_SUBMIT": "1"})
    for value in ("", "0", "yes", "true"):
        assert not submit_enabled({"RCLUST_TEST_SUBMIT": value})


# --- never opening a connection --------------------------------------------------------------------

def _client(socket_exists=True):
    socket = Mock()
    socket.exists.return_value = socket_exists
    socket.__str__ = lambda self: "/tmp/sock"
    return SimpleNamespace(socket_path=socket, destination="hpc1.example.edu", name="cluster-a",
                           host="hpc1.example.edu",
                           _get_base_flags=lambda: ["ssh", "-o", "ConnectTimeout=10"],
                           connect=Mock(), execute_command=Mock(), rsync=Mock())


def test_master_open_without_socket_runs_nothing():
    with patch("integration.clusters.subprocess.run") as run:
        assert not master_open(_client(socket_exists=False))
    run.assert_not_called()


@pytest.mark.parametrize("code, is_open", [(0, True), (255, False)])
def test_master_open_uses_ssh_o_check_in_batch_mode(code, is_open):
    with patch("integration.clusters.subprocess.run",
               return_value=SimpleNamespace(returncode=code)) as run:
        assert master_open(_client()) is is_open
    cmd = run.call_args.args[0]
    assert cmd[:1] == ["ssh"] and "BatchMode=yes" in cmd
    assert cmd[cmd.index("-O") + 1] == "check" and cmd[-1] == "hpc1.example.edu"
    assert "-M" not in cmd and "-f" not in cmd


def test_guarded_ssh_refuses_to_log_in_or_run_without_open_master():
    client = _client(socket_exists=False)
    guarded = GuardedSSH(client)
    with pytest.raises(NotConnected, match="rclust connect cluster-a"):
        guarded.connect()
    with pytest.raises(NotConnected):
        guarded.execute_command("squeue")
    with pytest.raises(NotConnected):
        guarded.rsync("a", "b")
    client.connect.assert_not_called()
    client.execute_command.assert_not_called()
    client.rsync.assert_not_called()


def test_guarded_ssh_uses_open_master_and_logs():
    client = _client()
    client.execute_command.return_value = (0, "out", "")
    guarded = GuardedSSH(client)
    with patch("integration.clusters.subprocess.run", return_value=SimpleNamespace(returncode=0)):
        guarded.connect()
        assert guarded.execute_command("sinfo", use_login_shell=True) == (0, "out", "")
    client.connect.assert_not_called()
    assert guarded.log == [("sinfo", 0, "out", "")]


# --- pytest behaviour, in a child pytest with a fake `ssh` ------------------------------------------

FAKE_SSH = """#!/bin/sh
echo "$*" >> "$FAKE_SSH_LOG"
for arg in "$@"; do
  case "$arg" in
    -G) last=""; for a in "$@"; do last=$a; done
        printf 'hostname %s.example.edu\\nuser someone\\nport 22\\ncontrolpath %s/%%n.sock\\n' \\
          "$last" "$FAKE_SSH_SOCKETS"; exit 0 ;;
    check) echo "Control socket connect: Connection refused" >&2; exit 255 ;;
  esac
done
echo "fake ssh: unexpected call" >&2; exit 255
"""


@pytest.fixture
def child_pytest(tmp_path):
    """Run pytest on tests/integration with a placeholder config and a fake ssh on PATH."""
    bin_dir, sockets = tmp_path / "bin", tmp_path / "sockets"
    bin_dir.mkdir()
    sockets.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
    for name in ("cluster-a", "cluster-b"):  # a stale socket file: `ssh -O check` is really asked
        (sockets / f"{name}.sock").touch()
    config = tmp_path / "config.yaml"
    config.write_text("clusters:\n"
                      "  cluster-a:\n    host: cluster-a\n    resources:\n      gpus: [h100]\n"
                      "  cluster-b:\n    host: cluster-b\n")
    log = tmp_path / "ssh.log"
    log.touch()

    def run(extra_path=None, **env_vars):
        env = {k: v for k, v in os.environ.items() if not k.startswith("RCLUST_")}
        path = [str(bin_dir)] + ([str(extra_path)] if extra_path else []) + [env["PATH"]]
        env.update(PATH=os.pathsep.join(path), RCLUST_CONFIG=str(config),
                   FAKE_SSH_LOG=str(log), FAKE_SSH_SOCKETS=str(sockets),
                   XDG_DATA_HOME=str(tmp_path / "data"), **env_vars)
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/integration", "-v", "-rs",
                                 "-p", "no:cacheprovider"],
                                cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
        calls = log.read_text().splitlines()
        return result, calls

    return run


def test_skipped_by_default_without_ssh_or_submit(child_pytest):
    result, calls = child_pytest()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "9 skipped" in result.stdout and "passed" not in result.stdout
    assert "test_submit" not in result.stdout  # not even collected
    assert calls == []  # the config is not read and no ssh runs


def test_cluster_without_open_master_is_skipped(child_pytest):
    result, calls = child_pytest(RCLUST_TEST_CLUSTERS="all")
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("cluster-a", "cluster-b"):
        assert f"run `rclust connect {name}` first" in result.stdout
    assert "passed" not in result.stdout and "failed" not in result.stdout
    assert "test_submit" not in result.stdout
    checks = [c for c in calls if "-O check" in c]
    assert checks and all("BatchMode=yes" in c for c in checks)
    assert all("-G" in c.split() or "-O check" in c for c in calls), calls  # never a login


def test_submit_test_needs_its_own_opt_in(child_pytest):
    result, _ = child_pytest(RCLUST_TEST_CLUSTERS="cluster-b", RCLUST_TEST_SUBMIT="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_submit.py::test_tiny_job_completes[cluster-b] SKIPPED" in result.stdout
    assert "spends a little of your allocation" in result.stdout

    result, _ = child_pytest(RCLUST_TEST_SUBMIT="1")  # without clusters: still nothing runs
    assert result.returncode == 0
    assert "test_tiny_job_completes[not-opted-in] SKIPPED" in result.stdout


def test_unknown_cluster_fails_listing_configured(child_pytest):
    result, calls = child_pytest(RCLUST_TEST_CLUSTERS="cluster-a,cluster-z")
    assert result.returncode == 4
    output = result.stdout + result.stderr
    assert "cluster-z" in output and "Configured: cluster-a, cluster-b" in output
    assert calls == []


# --- the real-cluster tests themselves, against a fake cluster -------------------------------------

FAKE_OPEN_SSH = """#!/bin/sh
echo "$*" >> "$FAKE_SSH_LOG"
for arg in "$@"; do
  case "$arg" in
    -G) last=""; for a in "$@"; do last=$a; done
        printf 'hostname %s.example.edu\\nuser someone\\nport 22\\ncontrolpath %s/%%n.sock\\n' \\
          "$last" "$FAKE_SSH_SOCKETS"; exit 0 ;;
    check) exit 0 ;;
  esac
done
for last in "$@"; do :; done
# the "cluster" runs the command locally, in a time zone that is not a whole number of hours away
TZ=Asia/Kolkata exec sh -c "$last"
"""

FAKE_SLURM = {
    "sinfo": 'case "$*" in *%1000G*) echo "gpu:h100:4(S:0-1)"; echo "(null)";; *) echo "p1 up";; esac',
    "squeue": "exit 0",
    "sshare": "echo 0.8",
    "sacct": 'now=$(date +%Y-%m-%dT%H:%M:%S); echo "7|$now|$now|Unknown|RUNNING|p1|cpu=1,mem=1G,node=1|00:10:00|00:00:05"',
    "sbatch": 'echo "sbatch: Job 99 to start at $(date +%Y-%m-%dT%H:%M:%S) using 1 processors on nodes n1 in partition p1" >&2',
    "base64": 'exec "{python}" -c "import base64,sys; sys.stdout.write(base64.b64encode(sys.stdin.buffer.read()).decode())"',
}


def test_read_only_tests_pass_against_a_fake_cluster(child_pytest, tmp_path):
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name, body in FAKE_SLURM.items():
        path = slurm / name
        path.write_text("#!/bin/sh\n" + body.replace("{python}", sys.executable) + "\n")
        path.chmod(0o755)
    (tmp_path / "bin" / "ssh").write_text(FAKE_OPEN_SSH)
    home = tmp_path / "home"
    home.mkdir()
    result, calls = child_pytest(extra_path=slurm, RCLUST_TEST_CLUSTERS="cluster-a", HOME=str(home))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "9 passed" in result.stdout, result.stdout
    assert not any(c.split()[-1] == "cluster-a" and "-O" not in c and "-G" not in c for c in calls)
