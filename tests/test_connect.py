"""`rclust connect` defers to the user's ssh config for ControlPersist unless -p is given."""

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
        SSHClient("cluster-a").connect(**kwargs)
    (master,) = [c for c in calls if "-M" in c]
    return master


def test_connect_without_persist_uses_ssh_config():
    master = _connect_args()
    assert not any(a.startswith("ControlPersist") for a in master)


@pytest.mark.parametrize("value", ["4h", "30m"])
def test_connect_with_persist_sets_control_persist(value):
    master = _connect_args(persist=value)
    assert f"ControlPersist={value}" in master
