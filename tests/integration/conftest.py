"""Opt-in real-cluster tests: parametrized by the clusters in RCLUST_TEST_CLUSTERS, skipped otherwise.

See clusters.py for how clusters are chosen and why no test ever opens a new SSH connection.
"""

import pytest

from .clusters import GuardedSSH, SKIP_REASON, connect_hint, master_open, selection, submit_enabled


def pytest_ignore_collect(collection_path, config):
    # The job-submitting test is not even collected without RCLUST_TEST_SUBMIT=1.
    if (collection_path.name == "test_submit.py" and collection_path.parent.name == "integration"
            and not submit_enabled()):
        return True
    return None


def pytest_generate_tests(metafunc):
    if "cluster_name" not in metafunc.fixturenames:
        return
    _, names = selection(metafunc.config)
    if names:
        metafunc.parametrize("cluster_name", names, ids=names, scope="session")
    else:
        metafunc.parametrize("cluster_name", [pytest.param(None, marks=pytest.mark.skip(reason=SKIP_REASON))],
                             ids=["not-opted-in"], scope="session")


def pytest_collection_modifyitems(config, items):
    # Belt and braces: nothing marked `cluster` runs unless clusters were selected.
    if selection(config)[1]:
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.get_closest_marker("cluster"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def rclust_config(request):
    rclust_config, names = selection(request.config)
    if not names:
        pytest.skip(SKIP_REASON)
    return rclust_config


@pytest.fixture(scope="session")
def ssh_clients(rclust_config):
    """One GuardedSSH per configured cluster (created lazily), shared by the whole run."""
    from rclust.ssh import SSHClient

    by_name = {c.name: c for c in rclust_config.clusters}
    clients = {}

    def provider(name):
        if name not in clients:
            c = by_name[name]
            clients[name] = GuardedSSH(SSHClient(c.host, c.user, c.ssh_key, name=c.name))
        return clients[name]

    return provider


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    """Keep rclust's history DB and watched-job state out of the user's own files."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    return tmp_path


@pytest.fixture
def scheduler(rclust_config, ssh_clients, tmp_path):
    from rclust.api import Scheduler

    return Scheduler(str(rclust_config.path), project_dir=tmp_path / "state", ssh_provider=ssh_clients)


@pytest.fixture
def cluster(cluster_name, scheduler, ssh_clients):
    """The cluster under test; skipped if its shared SSH connection is not open."""
    if not master_open(ssh_clients(cluster_name)._client):
        pytest.skip(connect_hint(cluster_name))
    return scheduler.get_cluster_by_name(cluster_name)


@pytest.fixture
def run_rclust(rclust_config, ssh_clients, tmp_path, monkeypatch):
    """Run the `rclust` CLI in-process against the selected config, with guarded SSH."""
    from click.testing import CliRunner

    import rclust.api
    from rclust.cli import main

    real_scheduler = rclust.api.Scheduler

    def scheduler_factory(config_path=None, project_dir=None, ssh_provider=None):
        return real_scheduler(config_path, project_dir=tmp_path / "state", ssh_provider=ssh_clients)

    monkeypatch.setattr(rclust.api, "Scheduler", scheduler_factory)

    def run(*args):
        return CliRunner().invoke(main, ["--config", str(rclust_config.path), *args])

    return run
