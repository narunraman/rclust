from typing import Optional, List, Callable, Tuple, Any
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from pathlib import Path
import os
import logging

from .config import GlobalConfig
from .dispatcher import Dispatcher
from .cluster import JobSpec, Cluster, ClusterMetrics
from .parser import JobParser
from .watcher import JobWatcher, WatchedJob

logger = logging.getLogger(__name__)

SSHProvider = Callable[[str], Any]  # cluster name -> SSH client with execute_command()


@dataclass
class JobHandle:
    """Handle to a submitted job for tracking and waiting."""

    job_id: str
    cluster: Cluster
    job_spec: JobSpec
    estimated_start: Optional[datetime] = None
    status: str = "SUBMITTED"
    node: Optional[str] = None

    @property
    def cluster_name(self) -> str:
        return self.cluster.name

    @property
    def estimated_wait(self) -> Optional[timedelta]:
        """Estimated wait time from now."""
        if self.estimated_start:
            return self.estimated_start - datetime.now()
        return None


class Scheduler:
    """High-level API for standalone use or integration with other SSH clients.

    Can operate in two modes:
    1. Standalone: Creates its own SSH connections (for `rclust` CLI)
    2. Library: Uses injected SSH clients supplied by the caller.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        project_dir: Optional[Path] = None,
        ssh_provider: Optional[SSHProvider] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            config_path: Path to config.yaml. If None, the first that exists of
                         $RCLUST_CONFIG, ./config.yaml, ~/.config/rclust/config.yaml.
            project_dir: Directory for watched-job state (default ./.cluster-scheduler)
            ssh_provider: Optional function(cluster_name) -> SSH client.
                         If provided, uses external SSH connections instead
                         of creating internal ones. Useful for integration
                         with applications that manage their own SSH sessions.
        """
        self.config = GlobalConfig.load(config_path)
        self.ssh_provider = ssh_provider
        self.dispatcher = Dispatcher(self.config, ssh_provider=ssh_provider)
        self.project_dir = (
            Path(project_dir) if project_dir else Path(".cluster-scheduler")
        )
        self._watcher: Optional[JobWatcher] = None

    @property
    def watcher(self) -> JobWatcher:
        """Lazy-initialize the job watcher."""
        if self._watcher is None:
            self._watcher = JobWatcher(
                project_dir=self.project_dir,
                clusters={c.name: c for c in self.dispatcher.clusters},
            )
        return self._watcher

    def analyze(self, target: str) -> JobSpec:
        """
        Analyzes a target (file or directory) to determine job requirements.
        """
        if os.path.isfile(target):
            return JobParser.parse_file(target)
        elif os.path.isdir(target):
            return JobSpec()  # a farm directory: no #SBATCH header to read
        else:
            raise ValueError(f"Target not found: {target}")

    def select_cluster(
        self,
        job_spec: JobSpec | dict[str, JobSpec] | None = None,
        policy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Tuple[Optional[Cluster], Optional[ClusterMetrics]]:
        """
        Select best cluster and return with metrics.

        Args:
            job_spec: Job requirements. Can be:
                     - JobSpec: Used for all clusters
                     - dict[str, JobSpec]: Per-cluster specs (cluster not in dict is excluded)
                     - None: Uses default minimal spec
            policy: Selection policy ("earliest" or "balanced"); None uses the config default
            tags: Required cluster tags
            exclude: Cluster names to exclude

        Returns:
            Tuple of (Cluster, ClusterMetrics) or (None, None) if no match
        """
        if job_spec is None:
            job_spec = JobSpec(cpus=1, time="1:00:00")

        cluster = self.dispatcher.propose_cluster(
            job_spec,
            policy_name=policy,
            required_tags=tags,
            exclude_clusters=exclude,
        )

        if cluster:
            return cluster, self.dispatcher.last_metrics.get(cluster)

        return None, None

    def get_queue_estimate(
        self,
        job_spec: Optional[JobSpec] = None,
        policy: Optional[str] = None,
        exclude: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[timedelta]]:
        """
        Get estimated wait time for the best cluster.

        Returns:
            Tuple of (cluster_name, wait_time) or (None, None)
        """
        cluster, metrics = self.select_cluster(job_spec, policy, exclude=exclude)

        if cluster and metrics:
            wait = metrics.estimated_start_time - datetime.now()
            return cluster.name, wait

        return None, None

    def propose_cluster(
        self,
        job_spec: Optional[JobSpec] = None,
        policy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Optional[Cluster]:
        """Select best cluster (legacy method, use select_cluster for metrics)."""
        if job_spec is None:
            job_spec = JobSpec(cpus=1, time="1:00:00")
        return self.dispatcher.propose_cluster(
            job_spec,
            policy_name=policy,
            required_tags=tags,
            exclude_clusters=exclude,
        )

    def submit_job(
        self,
        cluster: Cluster,
        script_path: str,
        job_spec: JobSpec,
        remote_dir: Optional[str] = None,
    ) -> str:
        return cluster.submit_job(script_path, job_spec, remote_dir=remote_dir)

    def submit_farm(self, cluster: Cluster, farm_dir: str) -> str:
        return cluster.submit_farm(farm_dir)

    def submit(
        self,
        script_path: str,
        job_spec: Optional[JobSpec] = None,
        cluster_name: Optional[str] = None,
        policy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        remote_dir: Optional[str] = None,
        after_any: Optional[List[str]] = None,
        after_all: Optional[List[str]] = None,
    ) -> JobHandle:
        """
        Submit a job and return a handle for tracking.

        Args:
            script_path: Path to SLURM script
            job_spec: Job requirements (parsed from script if not provided)
            cluster_name: Explicit cluster (bypasses selection)
            policy: Selection policy if cluster not specified
            tags: Required cluster tags
            exclude: Clusters to exclude
            remote_dir: Remote working directory
            after_any: Start once all these jobs have ended, in any state (afterany)
            after_all: Start once all these jobs have completed successfully (afterok)

        Returns:
            JobHandle for tracking the job
        """
        if job_spec is None:
            job_spec = self.analyze(script_path)
        else:
            job_spec = replace(job_spec)

        if after_any and after_all:
            raise ValueError("after_any and after_all cannot be used together")
        if after_any:
            job_spec = replace(job_spec, dependency=f"afterany:{':'.join(after_any)}")
        elif after_all:
            job_spec = replace(job_spec, dependency=f"afterok:{':'.join(after_all)}")

        # probe with at least 1 CPU for an hour
        if not job_spec.cpus and not job_spec.gpus:
            job_spec.cpus = 1
        if not job_spec.time:
            job_spec.time = "1:00:00"

        metrics = None
        if cluster_name:
            cluster = self.dispatcher.get_cluster_by_name(cluster_name)
            if not cluster:
                raise ValueError(f"Unknown cluster: {cluster_name}")
            metrics = self.dispatcher.last_metrics.get(cluster)
        else:
            cluster, metrics = self.select_cluster(job_spec, policy, tags, exclude)
            if not cluster:
                raise RuntimeError("No suitable cluster found")

        result = cluster.submit_job(script_path, job_spec, remote_dir=remote_dir)
        job_id = self._parse_job_id(result)

        return JobHandle(
            job_id=job_id,
            cluster=cluster,
            job_spec=job_spec,
            estimated_start=metrics.estimated_start_time if metrics else None,
        )

    def wait_until(
        self,
        job: JobHandle,
        condition: Optional[Callable[[WatchedJob], bool]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for a job to meet a condition.

        Args:
            job: Job handle from submit()
            condition: Custom condition function(WatchedJob) -> bool.
                       If None, waits for SLURM RUNNING.
            timeout: Max seconds to wait. If None, uses adaptive timeout
                     based on estimated_start + buffer.

        Returns:
            True if condition was met, False if timeout/failure
        """
        if timeout is None and job.estimated_start:
            # estimated wait + 50% + 10 min
            wait_estimate = (job.estimated_start - datetime.now()).total_seconds()
            timeout = max(wait_estimate * 1.5 + 600, 600)
        elif timeout is None:
            timeout = 3600  # 1 hour default
        if timeout <= 0:
            return False

        watched = WatchedJob(
            job_id=job.job_id,
            cluster_name=job.cluster_name,
            estimated_start=job.estimated_start,
        )

        ready_event = {"done": False, "success": False}

        def on_ready(w):
            ready_event["done"] = True
            ready_event["success"] = True

        watcher = JobWatcher(project_dir=self.project_dir, clusters={job.cluster_name: job.cluster},
                             load_saved=False, persist=False)
        watcher.add_job(watched, on_running=on_ready, condition=condition)
        import threading

        timer = threading.Timer(timeout, watcher.stop)
        timer.daemon = True
        timer.start()
        try:
            watcher.start(blocking=True)
        finally:
            timer.cancel()

        return ready_event["success"]

    def submit_async(
        self,
        script_path: str,
        job_spec: Optional[JobSpec] = None,
        on_running: Optional[Callable[[WatchedJob], None]] = None,
        condition: Optional[Callable[[WatchedJob], bool]] = None,
        **kwargs,
    ) -> JobHandle:
        """
        Submit a job and return immediately. Callback fires when ready.

        Args:
            script_path: Path to SLURM script
            job_spec: Job requirements
            on_running: Callback when job is RUNNING (or condition met)
            condition: Custom condition function
            **kwargs: Passed to submit()

        Returns:
            JobHandle for the submitted job
        """
        job = self.submit(script_path, job_spec, **kwargs)

        watched = WatchedJob(
            job_id=job.job_id,
            cluster_name=job.cluster_name,
            estimated_start=job.estimated_start,
        )

        self.watcher.add_job(watched, on_running=on_running, condition=condition)

        if self.watcher._thread is None or not self.watcher._thread.is_alive():
            self.watcher.start(blocking=False)

        return job

    def auto_submit(
        self,
        target: str,
        overrides: Optional[JobSpec] = None,
        policy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        remote_dir: Optional[str] = None,
    ) -> str:
        """
        High-level method to analyze, propose, and submit in one go.
        """
        spec = self.analyze(target)

        if overrides:
            for field in fields(JobSpec):
                value = getattr(overrides, field.name)
                if value is not None and field.name != "gpus_per_node":
                    setattr(spec, field.name, value)
            if overrides.memory is not None:
                spec.memory_per_cpu = None
            elif overrides.memory_per_cpu is not None:
                spec.memory = None
            if overrides.gpus is not None:
                if overrides.gpus_per_node:
                    spec.gpus_per_node = True
                if not overrides.gpus:
                    spec.gpu_type = None

        if not spec.cpus and not spec.gpus:
            spec.cpus = 1
        if not spec.time:
            spec.time = "1:00:00"

        cluster = self.propose_cluster(spec, policy=policy, tags=tags, exclude=exclude)

        if not cluster:
            raise RuntimeError("No suitable cluster found.")

        if os.path.isdir(target):
            return self.submit_farm(cluster, target)
        else:
            return self.submit_job(cluster, target, spec, remote_dir=remote_dir)

    def _parse_job_id(self, sbatch_output: str) -> str:
        """Extract job ID from sbatch output."""
        # Output is typically "Submitted batch job 12345"
        parts = sbatch_output.strip().split()
        if len(parts) >= 4 and parts[-1].isdigit():
            return parts[-1]
        for part in parts:
            if part.isdigit():
                return part
        raise ValueError(f"Could not parse job ID from: {sbatch_output}")

    def get_cluster_by_name(self, name: str) -> Optional[Cluster]:
        """Get a cluster by name."""
        return self.dispatcher.get_cluster_by_name(name)

    def list_clusters(self) -> List[str]:
        """List all configured cluster names."""
        return [c.name for c in self.dispatcher.clusters]
