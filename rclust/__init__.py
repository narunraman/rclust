"""Send a Slurm job to whichever of your clusters will start it soonest.

The public names below are imported on first use, so `import rclust.cli` (and `rclust --help`)
does not pay for modules a command doesn't need.
"""

import importlib

_EXPORTS = {
    "Scheduler": "api",
    "JobHandle": "api",
    "JobSpec": "cluster",
    "Cluster": "cluster",
    "ClusterMetrics": "cluster",
    "GlobalConfig": "config",
    "ClusterConfig": "config",
    "Policy": "dispatcher",
    "QueueTimePolicy": "dispatcher",
    "FastestStartPolicy": "dispatcher",
    "BalancedPolicy": "dispatcher",
    "JobWatcher": "watcher",
    "WatchedJob": "watcher",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
