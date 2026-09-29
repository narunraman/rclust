"""Old name for the `rclust` package, kept so older integrations that `import scheduler` keep working."""

import importlib
import sys
import warnings

warnings.warn("the `scheduler` package is now `rclust`; import from rclust instead",
              DeprecationWarning, stacklevel=2)

from rclust import *  # noqa: E402,F401,F403
from rclust import __all__  # noqa: E402,F401

for _name in ("api", "cli", "cluster", "config", "dispatcher", "history", "parser", "ssh", "watcher"):
    globals()[_name] = sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"rclust.{_name}")
