"""`rclust connect` defers to the user's ssh config for ControlPersist unless -p is given."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rclust.ssh import SSHClient


def _connect_args(**kwargs):
    calls = []

    def run(cmd, **_):
        calls.append(cmd)
        if "-G" in cmd:
            return SimpleNamespace(returncode=0, stdout="hostname h\ncontrolpath none\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("rclust.ssh.subprocess.run", side_effect=run), \
            patch("pathlib.Path.mkdir"), patch("pathlib.Path.chmod"), \
            patch("pathlib.Path.is_symlink", return_value=False), \
            patch("pathlib.Path.exists", return_value=False), \
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_uid=__import__("os").getuid())):
        client = SSHClient("cluster-a")
        with patch.object(SSHClient, "is_connected", side_effect=[False, True]):  # open after connecting
            client.connect(**kwargs)
    (master,) = [c for c in calls if "-M" in c]
    return master


def test_connect_without_persist_uses_ssh_config():
    master = _connect_args()
    assert not any(a.startswith("ControlPersist") for a in master)


@pytest.mark.parametrize("value", ["4h", "30m"])
def test_connect_with_persist_sets_control_persist(value):
    master = _connect_args(persist=value)
    assert f"ControlPersist={value}" in master


def test_connect_keeps_the_connection_alive():
    # a dead network is noticed (and the master closed) instead of commands hanging
    master = _connect_args()
    assert "ServerAliveInterval=15" in master and "ServerAliveCountMax=4" in master


def test_connect_without_prompts_says_how_to_log_in():
    from rclust.ssh import NotConnectedError

    def run(cmd, **kwargs):
        if "-G" in cmd:
            return SimpleNamespace(returncode=0, stdout="hostname h\ncontrolpath none\n", stderr="")
        kwargs["stderr"].write("Permission denied (keyboard-interactive).\n")
        return SimpleNamespace(returncode=255, stdout="", stderr="")

    with patch("rclust.ssh.subprocess.run", side_effect=run), patch("pathlib.Path.mkdir"), \
            patch("pathlib.Path.chmod"), patch("pathlib.Path.is_symlink", return_value=False), \
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_uid=__import__("os").getuid())):
        client = SSHClient("h.example.org", name="cluster-a")
        client.connect_command = "mytool connect"
        with patch.object(SSHClient, "is_connected", return_value=False), \
                pytest.raises(NotConnectedError, match="Could not connect to cluster-a .*Permission denied.*Run `mytool connect cluster-a`"):
            client.connect(interactive=False)


def _client_with(results):
    """A client whose ssh calls return `results` in turn (an exception is raised)."""
    calls = []

    def run(cmd, **kwargs):
        if "-G" in cmd:
            return SimpleNamespace(returncode=0, stdout="hostname h\ncontrolpath none\n", stderr="")
        calls.append((cmd, kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(returncode=result[0], stdout=result[1], stderr=result[2])

    patcher = patch("rclust.ssh.subprocess.run", side_effect=run)
    patcher.start()
    return SSHClient("h.example.org", name="cluster-a"), calls, patcher


def test_login_shell_output_drops_what_the_profile_prints_and_input_goes_to_stdin():
    client, calls, patcher = _client_with([(0, "Welcome to the cluster!\nRCLUST_OUTPUT_START\nhello\n", "")])
    try:
        assert client.execute_command("cat", use_login_shell=True, input="secret") == (0, "hello", "")
    finally:
        patcher.stop()
    cmd, kwargs = calls[0]
    assert cmd[-1] == "bash -l -c 'echo RCLUST_OUTPUT_START; cat'"
    assert "ControlMaster=no" in cmd  # never starts a master of its own
    assert kwargs["input"] == "secret"  # never on the command line


def test_a_slow_command_is_a_timeout():
    client, calls, patcher = _client_with([subprocess.TimeoutExpired("ssh", 15), (0, "", "")])
    try:
        assert client.execute_command("squeue", timeout=15) == (-1, "", "SSH command timed out")
    finally:
        patcher.stop()
    assert calls[1][0][-1] == "true"  # the connection was checked, and answered


def test_a_connection_that_stopped_responding_is_closed_and_reported():
    client, calls, patcher = _client_with([subprocess.TimeoutExpired("ssh", 15), subprocess.TimeoutExpired("ssh", 10),
                                           (0, "", "")])
    try:
        code, _, err = client.execute_command("squeue", timeout=15)
    finally:
        patcher.stop()
    assert code == 255
    assert "not connected to cluster-a (the connection stopped responding and was closed)" in err
    assert "rclust connect cluster-a" in err
    assert ["-O", "exit"] == calls[2][0][3:5]  # the stuck master is closed
