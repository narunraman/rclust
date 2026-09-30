"""Offline tests for config loading, the ssh layer, script parsing, submission and the CLI."""

import shlex
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from rclust.api import JobHandle, Scheduler
from rclust.cli import main
from rclust.cluster import JobSpec, SlurmCluster, ClusterMetrics
from rclust.config import ClusterConfig, ConfigError, GlobalConfig
from rclust.dispatcher import Dispatcher
from rclust.parser import JobParser
from rclust.ssh import SSHClient, quote_remote_path
from rclust.watcher import JobWatcher, WatchedJob


def scheduler(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("clusters:\n  cluster-a:\n    host: cluster-a\n")
    return Scheduler(str(config), project_dir=tmp_path / "state")


@pytest.mark.parametrize("content", ["clusters: []", "clusters: {}", "clusters: [oops]",
                                      "clusters:\n  a:\n    resources: [invalid]",
                                      "clusters:\n  a:\n    user: 123",
                                      "clusters:\n  a:\n    tags: gpu",
                                      "clusters:\n  a:\n    gpu_types: wrong"])
def test_invalid_config_has_actionable_error(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError):
        GlobalConfig.load(str(path))


def test_config_defaults_to_ssh_alias_without_username(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("clusters:\n  cluster-a:\n")
    config = GlobalConfig.load(str(path))
    assert config.path == path
    assert config.clusters[0].host == "cluster-a"
    assert config.clusters[0].user is None


def test_ssh_uses_actual_alias_and_configured_user_port():
    with patch("rclust.ssh.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0,
            stdout="hostname login.example.org\nuser scientist\nport 2222\ncontrolpath ~/.ssh/cm-%r-%h-%p\n",
            stderr="")
        client = SSHClient("login-alias", name="label-only")
        assert run.call_args.args[0][-1] == "login-alias"
        assert client.destination == "login-alias"
        assert str(client.socket_path).endswith("cm-scientist-login.example.org-2222")
        run.reset_mock()
        client.execute_command("echo ready")
        run.assert_called_once()
        args = run.call_args.args[0]
        assert "BatchMode=yes" in args
        assert "StrictHostKeyChecking=no" not in args
        assert "None@" not in " ".join(args)


def test_shell_arguments_and_paths_remain_single_words():
    value = "a b;$(touch unexpected)"
    assert shlex.split(JobSpec(memory=value, dependency=value).to_sbatch_args()) == [
        f"--mem={value}", f"--dependency={value}"]
    assert shlex.split(quote_remote_path("/jobs/" + value)) == ["/jobs/" + value]
    assert quote_remote_path("~/" + value).startswith('"$HOME"/')


def test_parser_reads_full_header_and_stops_at_body():
    spec = JobParser.parse_content("""#!/bin/bash
#SBATCH -c8 --time=3:00:00 --nodes=2
#SBATCH --gres=gpu:h100:4 --partition='gpu-long'
#SBATCH --ntasks=2 --mem-per-cpu=4G
hostname
#SBATCH --gpus=99
""")
    assert (spec.cpus, spec.gpus, spec.gpu_type, spec.gpus_per_node) == (8, 4, "h100", True)
    assert (spec.nodes, spec.tasks, spec.partition, spec.memory_per_cpu) == (2, 2, "gpu-long", "4G")


def test_submission_uses_same_gpu_mapping_as_probe_and_checks_copy(tmp_path):
    ssh = Mock()
    ssh.execute_command.return_value = (0, "Submitted batch job 42", "")
    ssh.rsync.return_value = True
    config = ClusterConfig("cluster-a", "cluster-a", account="project account",
                           resources={"gpu_types": [{"type": "h100", "slurm_spec": "site-h100"}]})
    cluster = SlurmCluster(config, ssh_provider=lambda _: ssh)
    spec = JobSpec(gpus=2, gpu_type="h100", nodes=1, time="1:00:00")
    cluster._get_estimated_start_time(spec)
    cluster.submit_job(str(tmp_path / "job script.sh"), spec, remote_dir="~/jobs with spaces")
    commands = [call.args[0] for call in ssh.execute_command.call_args_list]
    assert "--gpus=site-h100:2" in commands[0]
    assert "--gpus=site-h100:2" in commands[-1]
    assert "'--account=project account'" in commands[-1]
    assert shlex.split(commands[-1])[-2:] == ["--", "$HOME/jobs with spaces/" + commands[-1].rsplit("/", 1)[1].rstrip("'")]
    ssh.execute_command.reset_mock()
    ssh.rsync.return_value = False
    with pytest.raises(RuntimeError, match="upload"):
        cluster.submit_job("job.sh", spec)
    # Never submit after a failed upload.
    assert not any("sbatch" in call.args[0] for call in ssh.execute_command.call_args_list)


def test_failed_probe_cannot_be_mistaken_for_start_now():
    ssh = Mock()
    ssh.execute_command.return_value = (1, "sbatch: Job 1 to start at 2026-01-01T12:00:00", "")
    cluster = SlurmCluster(ClusterConfig("a", "a"), ssh_provider=lambda _: ssh)
    assert cluster._get_estimated_start_time(JobSpec()) is None


def test_selection_probes_once_and_excludes_empty_mapping(tmp_path):
    api = scheduler(tmp_path)
    cluster = api.dispatcher.clusters[0]
    metrics = ClusterMetrics(0, 1, datetime.now())
    cluster.get_metrics = Mock(return_value=metrics)
    selected, estimate = api.select_cluster(JobSpec(cpus=2))
    assert selected == cluster and estimate == metrics
    cluster.get_metrics.assert_called_once()
    cluster.get_metrics.reset_mock()
    assert api.select_cluster({}) == (None, None)
    cluster.get_metrics.assert_not_called()


def test_filter_rejects_wrong_gpu_type_without_network():
    cluster = SlurmCluster(ClusterConfig("a", "a", resources={"gpus": ["v100"]}))
    dispatcher = Dispatcher(GlobalConfig([cluster.config]))
    assert not dispatcher.check_resources(cluster, JobSpec(gpus=1, gpu_type="h100"))


def test_watcher_queries_once_then_uses_accounting_for_finished_job(tmp_path):
    ssh = Mock()
    ssh.execute_command.side_effect = [(0, "", ""), (0, "COMPLETED", "")]
    watcher = JobWatcher(tmp_path, clusters={"a": SimpleNamespace(ssh=ssh)})
    watcher.add_job(WatchedJob("42", "a"))
    watcher.start(blocking=True)
    assert watcher.pending_count == 0
    assert ssh.execute_command.call_count == 2
    assert "squeue" in ssh.execute_command.call_args_list[0].args[0]
    assert "sacct" in ssh.execute_command.call_args_list[1].args[0]


def test_wait_until_respects_timeout_and_preserves_saved_jobs(tmp_path):
    api = scheduler(tmp_path)
    cluster = api.dispatcher.clusters[0]
    ssh = Mock()
    ssh.execute_command.return_value = (0, "PENDING (null)", "")
    cluster._ssh_provider = lambda _: ssh
    api.watcher.add_job(WatchedJob("saved", "cluster-a"))
    before = api.watcher.jobs_file.read_text()
    job = JobHandle("42", cluster, JobSpec())
    start = datetime.now()
    assert not api.wait_until(job, timeout=0.02)
    assert (datetime.now() - start).total_seconds() < 1
    assert api.watcher.jobs_file.read_text() == before


def test_cli_submits_parsed_requirements_on_selected_cluster(tmp_path):
    api = scheduler(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --gpus=2 --time=8:00:00 --nodes=1\ntrue\n")
    cluster = api.dispatcher.clusters[0]
    cluster._ssh_provider = lambda _: Mock()
    api.select_cluster = Mock(return_value=(cluster, ClusterMetrics(0, 1, datetime.now())))
    api.submit = Mock(return_value=JobHandle("42", cluster, JobSpec()))
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["submit", str(script), "--cpus", "4", "--after-any", "1,2"])
    assert result.exit_code == 0, result.output
    spec = api.select_cluster.call_args.args[0]
    assert (spec.cpus, spec.gpus, spec.time, spec.nodes) == (4, 2, "8:00:00", 1)
    assert spec.dependency == "afterany:1:2"
    assert api.submit.call_args.kwargs["cluster_name"] == "cluster-a"
    assert api.submit.call_args.kwargs["job_spec"] is spec


def test_cli_missing_config_exits_cleanly_without_traceback(tmp_path):
    result = CliRunner().invoke(main, ["--config", str(tmp_path / "missing.yaml"), "suggest"])
    assert result.exit_code == 1
    assert "no config file found" in result.output
    assert "Traceback" not in result.output


def test_cli_memory_override_replaces_per_cpu_request(tmp_path):
    api = scheduler(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --mem-per-cpu=4G\ntrue\n")
    cluster = api.dispatcher.clusters[0]
    cluster._ssh_provider = lambda _: Mock()
    api.select_cluster = Mock(return_value=(cluster, None))
    api.submit = Mock(return_value=JobHandle("42", cluster, JobSpec()))
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["submit", str(script), "--mem", "16G"])
    assert result.exit_code == 0, result.output
    spec = api.select_cluster.call_args.args[0]
    assert spec.memory == "16G" and spec.memory_per_cpu is None


def test_invalid_script_requirements_fail_before_ssh(tmp_path):
    api = scheduler(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --cpus-per-task=0\ntrue\n")
    cluster = api.dispatcher.clusters[0]
    client = Mock()
    cluster._ssh_provider = lambda _: client
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["submit", str(script)])
    assert result.exit_code == 1
    assert "invalid job requirements" in result.output
    client.connect.assert_not_called()


def test_gres_gpu_mapping_preserves_other_resources():
    spec = JobParser.parse_content("#SBATCH --gres=gpu:h100:2,scratch:1\n")
    assert spec.gres == "gpu:h100:2,scratch:1"
    spec.gpu_type = "site-h100"
    assert shlex.split(spec.to_sbatch_args()) == ["--gres=scratch:1,gpu:site-h100:2"]


def test_multiple_gpu_types_are_preserved():
    spec = JobParser.parse_content("#SBATCH --gres=gpu:a100:2,gpu:h100:1\n")
    assert spec.gpus == 3 and spec.gpu_type is None
    assert shlex.split(spec.to_sbatch_args()) == ["--gres=gpu:a100:2,gpu:h100:1"]


def test_watcher_keeps_identical_job_ids_on_different_clusters(tmp_path):
    ssh = Mock()
    ssh.execute_command.return_value = (0, "RUNNING node1", "")
    watcher = JobWatcher(tmp_path, clusters={name: SimpleNamespace(ssh=ssh) for name in ["a", "b"]})
    callbacks = []
    watcher.add_job(WatchedJob("42", "a"), on_running=lambda job: callbacks.append(job.cluster_name))
    watcher.add_job(WatchedJob("42", "b"), on_running=lambda job: callbacks.append(job.cluster_name))
    assert watcher.pending_count == 2
    watcher.start(blocking=True)
    assert callbacks == ["a", "b"]
    assert watcher.pending_count == 0


def test_cli_callback_waits_for_running_job_without_touching_saved_state(tmp_path):
    api = scheduler(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    cluster = api.dispatcher.clusters[0]
    client = Mock()
    client.execute_command.return_value = (0, "RUNNING node1", "")
    cluster._ssh_provider = lambda _: client
    api.select_cluster = Mock(return_value=(cluster, None))
    api.submit = Mock(return_value=JobHandle("42", cluster, JobSpec()))
    api.watcher.add_job(WatchedJob("unrelated", "cluster-a"))
    before = api.watcher.jobs_file.read_text()
    with patch("rclust.cli.get_scheduler", return_value=api), patch("rclust.cli.subprocess.run") as run:
        run.return_value.returncode = 0
        result = CliRunner().invoke(main, ["submit", str(script), "--on-running", "echo ready"])
    assert result.exit_code == 0, result.output
    run.assert_called_once_with("echo ready", shell=True)
    assert api.watcher.jobs_file.read_text() == before


def test_rsync_directory_copy_preserves_trailing_slash(tmp_path):
    with patch("rclust.ssh.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        client = SSHClient("cluster-a")
        client.rsync(str(tmp_path / "farm") + "/", "~/farms/example/")
        assert run.call_args.args[0][-2] == str(tmp_path / "farm") + "/"


def test_config_lookup_order(tmp_path, monkeypatch):
    from rclust.config import find_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("RCLUST_CONFIG", raising=False)
    assert find_config() is None
    user = tmp_path / "xdg" / "rclust" / "config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("clusters: {a: {}}")
    assert find_config() == user
    (tmp_path / "config.yaml").write_text("clusters: {b: {}}")
    assert find_config() == Path("config.yaml")
    env = tmp_path / "env.yaml"
    env.write_text("clusters: {c: {}}")
    monkeypatch.setenv("RCLUST_CONFIG", str(env))
    assert find_config() == env
    assert find_config(str(tmp_path / "explicit.yaml")) == tmp_path / "explicit.yaml"


def test_missing_config_says_how_to_create_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("RCLUST_CONFIG", raising=False)
    with pytest.raises(ConfigError, match="rclust config"):
        GlobalConfig.load()


def test_unconnected_cluster_error_says_to_run_connect():
    def run(cmd, **kwargs):
        if "-G" in cmd:
            return SimpleNamespace(returncode=0, stdout="hostname h\n", stderr="")
        if "-O" in cmd:  # control-socket check: no master running
            return SimpleNamespace(returncode=255, stdout="", stderr="")
        return SimpleNamespace(returncode=255, stdout="", stderr="Permission denied (keyboard-interactive).")
    with patch("rclust.ssh.subprocess.run", side_effect=run), patch("pathlib.Path.exists", return_value=True):
        code, _, err = SSHClient("h.example.org", name="cluster-a").execute_command("true")
    assert code == 255
    assert "not connected to cluster-a" in err and "rclust connect cluster-a" in err


def test_cli_explains_why_no_cluster_was_found(tmp_path):
    api = scheduler(tmp_path)
    cluster = api.dispatcher.clusters[0]
    ssh = Mock()
    ssh.execute_command.return_value = (255, "", "not connected to cluster-a (x); run `rclust connect cluster-a`")
    cluster._ssh_provider = lambda _: ssh
    with patch("rclust.cli.get_scheduler", return_value=api):
        result = CliRunner().invoke(main, ["suggest", "--gpus", "1"])
    assert result.exit_code == 1
    assert "cluster-a: not connected" in result.output and "rclust connect" in result.output
    assert "Traceback" not in result.output


def test_parser_reads_typed_gpu_requests():
    assert JobParser.parse_content("#SBATCH --gpus=a100:2\n").gpus == 2
    assert JobParser.parse_content("#SBATCH -G 3\n").gpus == 3
    assert JobParser.parse_content("#SBATCH --gres=gpu:1\n").gpus == 1
    assert JobParser.parse_content("## #SBATCH --gpus=4\n#SBATCH -c 2\n").gpus is None


def test_cli_policy_names_match_the_dispatcher():
    from rclust import cli

    assert set(cli.POLICIES) == set(Dispatcher.POLICIES)
    assert cli.POLICY_ALIASES == Dispatcher.ALIASES


def test_old_scheduler_import_still_works():
    import importlib
    import sys
    import warnings

    sys.modules.pop("scheduler", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module("scheduler")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    from scheduler.cluster import JobSpec as OldJobSpec
    assert old.Scheduler is Scheduler and OldJobSpec is JobSpec
