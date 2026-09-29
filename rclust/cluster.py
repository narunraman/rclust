from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Set
import re
import logging
import shlex
import uuid

from pathlib import Path
from .config import ClusterConfig
from .ssh import SSHClient, quote_remote_path

logger = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """Defines resources requested for a job."""

    cpus: Optional[int] = None
    gpus: Optional[int] = None
    gpu_type: Optional[str] = None  # e.g., "h100" - looked up in cluster's gpu_types
    memory: Optional[str] = None  # e.g., "16G"
    time: Optional[str] = None  # e.g., "2:00:00"
    dependency: Optional[str] = None  # SLURM dependency string (e.g., "afterany:123:456")
    nodes: Optional[int] = None  # --nodes=N; pin to a single node when GPUs must be co-located
    partition: Optional[str] = None
    tasks: Optional[int] = None
    memory_per_cpu: Optional[str] = None
    gpus_per_node: bool = False
    gres: Optional[str] = None  # Preserve other generic resources alongside GPU requests.

    def to_sbatch_args(self) -> str:
        if self.memory is not None and self.memory_per_cpu is not None:
            raise ValueError("memory and memory_per_cpu cannot be requested together")
        args = []
        for name in ("cpus", "gpus", "nodes", "tasks"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < (0 if name == "gpus" else 1)):
                raise ValueError(f"{name} must be a {'non-negative' if name == 'gpus' else 'positive'} integer")
        if self.cpus is not None:
            args.append(f"--cpus-per-task={self.cpus}")
        if self.gpus is not None:
            value = f"{self.gpu_type}:{self.gpus}" if self.gpu_type and self.gpus else str(self.gpus)
            if self.gres is not None:
                gpu_resources = [r for r in self.gres.split(",") if r.startswith("gpu:")]
                if len(gpu_resources) > 1:
                    original_count = sum(int(r.rsplit(":", 1)[1]) for r in gpu_resources)
                    if self.gpus != original_count or self.gpu_type:
                        raise ValueError("GPU count overrides require a single GPU type in --gres")
                    args.append(f"--gres={self.gres}")
                else:
                    resources = [r for r in self.gres.split(",") if r and not r.startswith("gpu:")]
                    resources.append(f"gpu:{value}")
                    args.append("--gres=" + ",".join(resources))
            else:
                option = "--gpus-per-node" if self.gpus_per_node else "--gpus"
                args.append(f"{option}={value}")
        elif self.gres:
            args.append(f"--gres={self.gres}")
        if self.memory:
            args.append(f"--mem={self.memory}")
        if self.time:
            args.append(f"--time={self.time}")
        if self.dependency:
            args.append(f"--dependency={self.dependency}")
        if self.nodes:
            args.append(f"--nodes={self.nodes}")
        if self.partition:
            args.append(f"--partition={self.partition}")
        if self.tasks:
            args.append(f"--ntasks={self.tasks}")
        if self.memory_per_cpu:
            args.append(f"--mem-per-cpu={self.memory_per_cpu}")
        return shlex.join(args)


@dataclass
class ClusterMetrics:
    load: int
    fairshare: float
    estimated_start_time: datetime


class Cluster:
    def __init__(self, config: ClusterConfig, ssh_provider=None):
        """
        Initialize cluster.

        Args:
            config: Cluster configuration
            ssh_provider: Optional function(cluster_name) -> SSH client.
                         If provided, uses external SSH instead of creating internal.
        """
        self.config = config
        self._ssh_provider = ssh_provider
        self._internal_ssh = None
        self.last_error: Optional[str] = None  # why the last probe failed

    @property
    def ssh(self):
        """Get SSH client (internal or external)."""
        if self._ssh_provider:
            return self._ssh_provider(self.name)
        if self._internal_ssh is None:
            self._internal_ssh = SSHClient(
                self.config.host, self.config.user, self.config.ssh_key,
                name=self.config.name
            )
        return self._internal_ssh

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tags(self) -> Set[str]:
        return self.config.tags

    @property
    def resources(self) -> dict:
        return self.config.resources

    @property
    def account(self) -> Optional[str]:
        return self.config.account

    @property
    def cpu_account(self) -> Optional[str]:
        return self.config.cpu_account

    def get_account_for_job(self, job_spec: JobSpec) -> Optional[str]:
        """Get the appropriate account for a job spec.
        
        Sites may use separate accounts for GPU and CPU jobs.
        Uses cpu_account for CPU-only jobs if configured, otherwise falls back to account.
        """
        if job_spec.gpus:
            return self.account
        return self.cpu_account or self.account

    @property
    def remote_dir(self) -> str:
        return self.config.remote_dir or "~/cluster_scheduler_jobs"

    def get_metrics(self, job_spec: JobSpec) -> Optional[ClusterMetrics]:
        raise NotImplementedError

    def submit_job(self, script_path: str, job_spec: JobSpec, remote_dir: Optional[str] = None) -> str:
        raise NotImplementedError

    def submit_farm(self, farm_dir: str) -> str:
        raise NotImplementedError


class SlurmCluster(Cluster):
    def _sbatch_args(self, job_spec: JobSpec) -> str:
        """Use the same safely quoted arguments for probes and actual submissions."""
        from dataclasses import replace

        gpu_type = job_spec.gpu_type
        if gpu_type:
            for gpu in self.resources.get("gpu_types", []):
                if gpu.get("type") == gpu_type:
                    gpu_type = gpu.get("slurm_spec") or gpu_type
                    break
        args = replace(job_spec, gpu_type=gpu_type).to_sbatch_args()
        account = self.get_account_for_job(job_spec)
        if account:
            args += " " + shlex.quote(f"--account={account}")
        return args

    def get_metrics(self, job_spec: JobSpec) -> Optional[ClusterMetrics]:
        """Probe the cluster. None if the start-time probe fails (reason in `last_error`).

        Fairshare and load only break ties and feed the balanced policy, so sites without
        sshare still work (fairshare 0.5, load 0).
        """
        self.last_error = None
        est_start = self._get_estimated_start_time(job_spec)
        if est_start is None:
            return None
        fs = self._get_fairshare()
        load = self._get_load()
        return ClusterMetrics(load=0 if load is None else load,
                              fairshare=0.5 if fs is None else fs,
                              estimated_start_time=est_start)

    def _get_fairshare(self) -> Optional[float]:
        code, out, _ = self.ssh.execute_command("sshare -u $USER -n -o LevelFS")
        if code != 0:
            return None
        values = []
        for line in out.split():
            try:
                values.append(float(line))
            except ValueError:
                pass
        # several lines when you belong to several accounts: the best one
        return max(values) if values else 0.5

    def discover_resources(self) -> dict:
        """GPU types on the cluster, from sinfo's GRES column, as {"gpus": [...]} (for `resources`)."""
        code, out, err = self.ssh.execute_command("sinfo -o '%N %P %G' --noheader")
        if code != 0:
            raise RuntimeError(err or f"sinfo failed with exit code {code}")
        gpus = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3 or "(null)" in parts[2]:
                continue
            for item in re.sub(r"\(.*?\)", "", parts[2]).split(","):  # drop "(S:0-3)" etc.
                sub = item.split(":")
                if sub[0] != "gpu" or len(sub) < 2:
                    continue
                if len(sub) == 2:            # gpu:COUNT
                    if sub[1].isdigit():
                        gpus.add("generic")
                elif not sub[1].isdigit():   # gpu:TYPE:COUNT
                    gpus.add(sub[1])
        return {"gpus": sorted(gpus)} if gpus else {}

    def _get_load(self) -> Optional[int]:
        code, out, _ = self.ssh.execute_command(
            "squeue -u $USER -t RUNNING --noheader | wc -l"
        )
        if code != 0:
            return None
        try:
            return int(out.strip())
        except ValueError:
            return 0

    def _get_estimated_start_time(self, job_spec: JobSpec) -> Optional[datetime]:
        """When `sbatch --test-only` says the job would start, in local time. None if the probe fails."""
        sbatch_args = self._sbatch_args(job_spec)

        # Also read the cluster's clock: --test-only answers in the cluster's local time, and clusters
        # may sit in different time zones. Keep sbatch's exit status so a failed probe cannot look
        # like "can start now".
        cmd = (f"sbatch --test-only {sbatch_args} --wrap='sleep 1'; rc=$?; "
               f"echo RCLUST_NOW $(date +%s) $(date +%Y-%m-%dT%H:%M:%S); exit $rc")

        # login shell, so site profile scripts (modules, default accounts) are loaded
        code, stdout, stderr = self.ssh.execute_command(cmd, use_login_shell=True)

        # sbatch prints its estimate on stderr
        output = stdout + "\n" + stderr
        if code != 0 or "error" in output.lower():
            detail = next((l.strip() for l in stderr.splitlines() if l.strip()), "") or f"exit code {code}"
            self.last_error = detail
            logger.info(f"Probe failed on {self.name}: {detail}")
            return None

        # "sbatch: Job 123 to start at 2025-12-15T12:00:00", in the cluster's local time
        output = re.sub(r"RCLUST_NOW \S+ \S+", "", output) if (clock := re.search(
            r"RCLUST_NOW (\d+) (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", output)) else output
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", output)
        if match:
            try:
                start = datetime.fromisoformat(match.group(1))
            except ValueError:
                start = None
            if start is not None:
                if clock:
                    # shift from the cluster's clock to ours
                    cluster_now = datetime.fromisoformat(clock.group(2))
                    offset = cluster_now - datetime.fromtimestamp(int(clock.group(1)))
                    start -= timedelta(minutes=round(offset.total_seconds() / 900) * 15)
                return start

        # A successful probe without a dated estimate is treated as immediate availability.
        if code == 0:
            return datetime.now()

        return None

    def submit_job(
        self, script_path: str, job_spec: JobSpec, remote_dir: Optional[str] = None
    ) -> str:
        timestamp = uuid.uuid4().hex
        base_dir = remote_dir or self.remote_dir

        script_name = Path(script_path).name

        remote_script_name = f"{timestamp}_{script_name}"
        remote_path = f"{base_dir}/{remote_script_name}"

        code, _, err = self.ssh.execute_command(f"mkdir -p -- {quote_remote_path(base_dir)}")
        if code != 0:
            raise RuntimeError(f"Could not create remote directory: {err}")
        if not self.ssh.rsync(script_path, remote_path):
            raise RuntimeError(f"Could not upload job script to {self.name}")

        # submitted from the home directory, so that is the job's working directory
        cmd = f"sbatch {self._sbatch_args(job_spec)} -- {quote_remote_path(remote_path)}"
        code, out, err = self.ssh.execute_command(cmd, use_login_shell=True)
        if code != 0:
            raise RuntimeError(err or f"sbatch failed with exit code {code}")
        return out.strip()  # "Submitted batch job 123"

    def submit_farm(self, farm_dir: str) -> str:
        """Copy a directory to ~/farms/<name> and run its ./submit.run (e.g. a META-Farm farm)."""
        dirname = Path(farm_dir).name
        remote_path = f"~/farms/{dirname}"
        code, _, err = self.ssh.execute_command(f"mkdir -p -- {quote_remote_path(remote_path)}")
        if code != 0:
            raise RuntimeError(f"Could not create farm directory: {err}")
        if not self.ssh.rsync(str(Path(farm_dir)) + "/", remote_path + "/"):
            raise RuntimeError(f"Failed to rsync farm directory to {self.name}")

        cmd = f"cd -- {quote_remote_path(remote_path)} && ./submit.run"
        code, out, err = self.ssh.execute_command(cmd)
        if code != 0:
            raise RuntimeError(f"./submit.run failed: {err}")
        return out.strip()
