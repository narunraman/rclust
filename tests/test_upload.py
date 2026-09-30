"""Uploads: rsync destinations that work with every rsync version, and clear failures."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from rclust.cluster import JobSpec, SlurmCluster
from rclust.config import ClusterConfig
from rclust.ssh import SSHClient, rsync_remote_path


@pytest.mark.parametrize("path, expected", [
    ("~/cluster_scheduler_jobs/abc_job.sh", "cluster_scheduler_jobs/abc_job.sh"),
    ("~/farms/example/", "farms/example/"),
    ("~", "."),
    ("~/", "."),
    ("/scratch/me/jobs", "/scratch/me/jobs"),
    ("jobs/x", "jobs/x"),
    ("-odd", "./-odd"),
])
def test_rsync_destination_never_needs_remote_shell_expansion(path, expected):
    assert rsync_remote_path(path) == expected


def _client_and_run(returncode=0, stderr=""):
    run = Mock(return_value=SimpleNamespace(returncode=returncode, stdout="", stderr=stderr))
    with patch("rclust.ssh.subprocess.run", run):
        client = SSHClient("cluster-a")
    return client, run


def test_rsync_default_remote_dir_is_relative_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RSYNC_PROTECT_ARGS", "1")
    client, run = _client_and_run()
    with patch("rclust.ssh.subprocess.run", run):
        assert client.rsync(str(tmp_path / "job.sh"), "~/cluster_scheduler_jobs/abc_job.sh")
    args, kwargs = run.call_args.args[0], run.call_args.kwargs
    assert args[-1] == "cluster-a:cluster_scheduler_jobs/abc_job.sh"
    assert "$HOME" not in " ".join(args) and "~" not in args[-1]
    # one quoting rule for old and new rsync: new rsync behaves like old, never with protect-args
    assert kwargs["env"]["RSYNC_OLD_ARGS"] == "1"
    assert "RSYNC_PROTECT_ARGS" not in kwargs["env"]


def test_rsync_quotes_paths_with_spaces_for_the_remote_shell(tmp_path):
    client, run = _client_and_run()
    with patch("rclust.ssh.subprocess.run", run):
        client.rsync(str(tmp_path / "job.sh"), "~/jobs with spaces/x.sh")
    assert run.call_args.args[0][-1] == "cluster-a:'jobs with spaces/x.sh'"


def test_failed_rsync_keeps_last_lines_of_stderr(tmp_path):
    stderr = "\n".join(f"line {i}" for i in range(6)) + "\nrsync error: some files could not be transferred (code 23)\n"
    client, run = _client_and_run(returncode=23, stderr=stderr)
    with patch("rclust.ssh.subprocess.run", run):
        assert not client.rsync(str(tmp_path / "job.sh"), "~/jobs/x.sh")
    assert client.last_rsync_error == "line 4\nline 5\nrsync error: some files could not be transferred (code 23)"


def _cluster(ssh):
    return SlurmCluster(ClusterConfig("cluster-a", "cluster-a"), ssh_provider=lambda _: ssh)


@pytest.mark.parametrize("existed", [False, True])
def test_failed_upload_reports_rsync_error_and_removes_only_a_new_directory(existed):
    ssh = Mock()
    ssh.execute_command.return_value = (0, "RCLUST_EXISTS" if existed else "", "")
    ssh.rsync.return_value = False
    ssh.last_rsync_error = 'rsync: change_dir#3 "/home/u/x" failed: No such file or directory (2)'
    with pytest.raises(RuntimeError) as info:
        _cluster(ssh).submit_job("job.sh", JobSpec(cpus=1, time="1:00:00"))
    message = str(info.value)
    assert "change_dir#3" in message
    commands = [c.args[0] for c in ssh.execute_command.call_args_list]
    assert "mkdir -p" in commands[0] and '"$HOME"/cluster_scheduler_jobs' in commands[0]
    assert not any("sbatch" in c for c in commands)
    rmdirs = [c for c in commands if c.startswith("rmdir")]
    if existed:
        assert not rmdirs and "removed" not in message
    else:
        assert rmdirs == ["rmdir -- \"$HOME\"/cluster_scheduler_jobs"]
        assert "removed the empty directory" in message


def test_farm_upload_uses_relative_destination(tmp_path):
    ssh = Mock()
    ssh.execute_command.return_value = (0, "done", "")
    ssh.rsync.return_value = True
    farm = tmp_path / "myfarm"
    farm.mkdir()
    assert _cluster(ssh).submit_farm(str(farm)) == "done"
    local, remote = ssh.rsync.call_args.args
    assert local == str(farm) + "/" and remote == "~/farms/myfarm/"
    assert rsync_remote_path(remote) == "farms/myfarm/"
