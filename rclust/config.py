import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


class ConfigError(ValueError):
    """The config file is missing or invalid."""


@dataclass
class ClusterConfig:
    name: str
    host: str
    user: Optional[str] = None      # None: whatever your SSH config says
    ssh_key: Optional[str] = None
    account: Optional[str] = None   # Slurm account (--account)
    cpu_account: Optional[str] = None  # separate account for CPU-only jobs, if your site uses one
    remote_dir: Optional[str] = None   # where job scripts are copied on the cluster
    tags: Set[str] = field(default_factory=set)
    # e.g. {"gpus": ["a100", "v100"], "gpu_types": [{"type": "h100", "slurm_spec": ...}]}
    resources: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Optional[dict]) -> "ClusterConfig":
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("cluster names must be non-empty strings")
        data = {} if data is None else data
        if not isinstance(data, dict):
            raise ConfigError(f"cluster '{name}': expected a mapping of settings")
        for key in ("host", "user", "ssh_key", "account", "cpu_account", "remote_dir"):
            value = data.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"cluster '{name}': {key} must be a non-empty string")
        resources = data.get("resources") or {}
        if not isinstance(resources, dict):
            raise ConfigError(f"cluster '{name}': resources must be a mapping")
        resources = dict(resources)
        tags = data.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            raise ConfigError(f"cluster '{name}': tags must be a list of strings")
        # `gpu_types` (with slurm_spec per type) may sit at the top level; it also implies `gpus`
        if "gpu_types" in data:
            resources["gpu_types"] = data["gpu_types"]
        gpu_types = resources.get("gpu_types") or []
        if not isinstance(gpu_types, list) or any(
            not isinstance(g, dict) or not isinstance(g.get("type"), str)
            or (g.get("slurm_spec") is not None and not isinstance(g["slurm_spec"], str))
            for g in gpu_types
        ):
            raise ConfigError(f"cluster '{name}': gpu_types must be a list of GPU definitions")
        if "gpu_types" in resources:
            if not resources.get("gpus"):
                gpus = [g["type"] for g in gpu_types]
                if gpus:
                    resources["gpus"] = gpus
        if "gpus" in resources and (not isinstance(resources["gpus"], list)
                                   or any(not isinstance(g, str) for g in resources["gpus"])):
            raise ConfigError(f"cluster '{name}': resources.gpus must be a list of strings")
        return cls(
            name=name,
            host=data.get("host") or name,  # default: the name is an alias from ~/.ssh/config
            user=data.get("user"),
            ssh_key=data.get("ssh_key"),
            account=data.get("account"),
            cpu_account=data.get("cpu_account"),
            remote_dir=data.get("remote_dir"),
            tags=set(tags),
            resources=resources,
        )


def user_config_path() -> Path:
    """~/.config/rclust/config.yaml, honouring $XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "rclust" / "config.yaml"


def config_search_paths() -> List[Path]:
    """Where rclust looks when no path is given, in order."""
    paths = []
    if os.environ.get("RCLUST_CONFIG"):
        paths.append(Path(os.environ["RCLUST_CONFIG"]).expanduser())
    paths += [Path("config.yaml"), user_config_path()]
    return paths


def find_config(path: Optional[str] = None) -> Optional[Path]:
    """The config file to use: an explicit path, else the first search path that exists."""
    if path:
        return Path(path).expanduser()
    return next((p for p in config_search_paths() if p.exists()), None)


@dataclass
class GlobalConfig:
    clusters: List[ClusterConfig]
    default_policy: str = "history"
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "GlobalConfig":
        """Load the config: `path` if given, else the first of $RCLUST_CONFIG, ./config.yaml and
        ~/.config/rclust/config.yaml that exists. Raises ConfigError if there is none or it is invalid."""
        import yaml

        config_path = find_config(path)
        if config_path is None or not config_path.exists():
            looked = [str(config_path)] if path else [str(p) for p in config_search_paths()]
            raise ConfigError(
                "no config file found (looked for " + ", ".join(looked) + "). "
                "Create one with `rclust config`, or copy config.yaml.example to "
                f"{user_config_path()}"
            )
        try:
            data = yaml.safe_load(config_path.read_text()) or {}
        except yaml.YAMLError:
            # YAML diagnostics may include private values from the configuration.
            raise ConfigError(f"{config_path} is not valid YAML") from None
        except OSError as e:
            raise ConfigError(f"Could not read config file: {config_path} ({e.strerror})") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{config_path}: expected a mapping with a 'clusters' key")
        cluster_data = data.get("clusters")
        if not isinstance(cluster_data, dict) or not cluster_data:
            raise ConfigError(f"{config_path}: 'clusters' must be a non-empty mapping")
        policy = data.get("default_policy") or "history"
        if policy not in ("history", "earliest", "balanced", "learned", "rush", "queue-time"):
            raise ConfigError(f"{config_path}: unknown default_policy")
        clusters = [ClusterConfig.from_dict(name, c) for name, c in cluster_data.items()]
        return cls(clusters=clusters, default_policy=policy,
                   path=config_path)
