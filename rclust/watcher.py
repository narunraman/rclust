"""Job watcher with adaptive polling based on estimated start times."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Any
import json
import threading
import logging
import shlex

from .cluster import Cluster

logger = logging.getLogger(__name__)


@dataclass
class WatchedJob:
    """A job being watched for status changes."""

    job_id: str
    cluster_name: str
    estimated_start: Optional[datetime] = None
    status: str = "PENDING"
    node: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "job_id": self.job_id,
            "cluster_name": self.cluster_name,
            "estimated_start": (
                self.estimated_start.isoformat() if self.estimated_start else None
            ),
            "status": self.status,
            "node": self.node,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchedJob":
        """Deserialize from dictionary."""
        est = data.get("estimated_start")
        return cls(
            job_id=data["job_id"],
            cluster_name=data["cluster_name"],
            estimated_start=datetime.fromisoformat(est) if est else None,
            status=data.get("status", "PENDING"),
            node=data.get("node"),
        )


class JobWatcher:
    """
    Watches jobs and triggers callbacks when conditions are met.

    Uses adaptive polling: checks less frequently for jobs with long
    estimated wait times, more frequently as the expected start approaches.
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        clusters: Optional[Dict[str, Cluster]] = None,
        load_saved: bool = True,
        persist: bool = True,
    ):
        """
        Initialize the job watcher.

        Args:
            project_dir: Directory for job state persistence.
                         Defaults to .cluster-scheduler/ in current directory.
            clusters: Dict of cluster_name -> Cluster instances for status checks.
        """
        self.project_dir = (
            Path(project_dir) if project_dir else Path(".cluster-scheduler")
        )
        self.jobs_file = self.project_dir / "jobs.json"
        self._save_state = persist
        self._watched: Dict[str, WatchedJob] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._custom_conditions: Dict[str, Callable] = {}
        self._clusters = clusters or {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load any persisted jobs
        if load_saved:
            self._load()

    def add_job(
        self,
        job: WatchedJob,
        on_running: Optional[Callable] = None,
        condition: Optional[Callable] = None,
    ):
        """
        Add a job to watch.

        Args:
            job: The job to watch
            on_running: Callback when job reaches RUNNING (or condition is met)
            condition: Optional custom condition function(WatchedJob) -> bool
                      If provided, callback fires when this returns True
                      while job is RUNNING
        """
        key = next((key for key, watched in self._watched.items()
                    if watched.job_id == job.job_id and watched.cluster_name == job.cluster_name),
                   job.job_id)
        existing = self._watched.get(key)
        if existing and existing.cluster_name != job.cluster_name:
            # Slurm job IDs can overlap between clusters.
            key = f"{job.cluster_name}:{job.job_id}"
        self._watched[key] = job
        if on_running:
            self._callbacks[key] = on_running
        if condition:
            self._custom_conditions[key] = condition
        self._persist()
        logger.info(f"Now watching job {job.job_id} on {job.cluster_name}")

    def remove_job(self, job_id: str):
        """Remove a job from the watch list."""
        self._watched.pop(job_id, None)
        self._callbacks.pop(job_id, None)
        self._custom_conditions.pop(job_id, None)
        self._persist()

    def get_poll_interval(self, job: WatchedJob) -> int:
        """
        Calculate adaptive poll interval based on expected wait time.

        Returns shorter intervals as the expected start time approaches.
        """
        if not job.estimated_start:
            return 60  # Default: 1 minute if no estimate

        wait_seconds = (job.estimated_start - datetime.now(job.estimated_start.tzinfo)).total_seconds()

        if wait_seconds < 0:
            # Should be running now, check frequently
            return 10
        elif wait_seconds < 300:  # < 5 min
            return 30
        elif wait_seconds < 1800:  # < 30 min
            return 120  # 2 minutes
        elif wait_seconds < 7200:  # < 2 hours
            return 300  # 5 minutes
        else:
            # Very long queue
            return 600  # 10 minutes

    def set_cluster(self, name: str, cluster: Cluster):
        """Register a cluster for status checks."""
        self._clusters[name] = cluster

    def start(self, blocking: bool = False):
        """
        Start watching jobs.

        Args:
            blocking: If True, runs in foreground. If False, spawns background thread.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        if blocking:
            self._watch_loop()
        else:
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()
            logger.info("Job watcher started in background")

    def stop(self):
        """Stop the watcher."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and self._thread != threading.current_thread():
            self._thread.join(timeout=5)
        logger.info("Job watcher stopped")

    def _watch_loop(self):
        """Main watch loop."""
        while not self._stop_event.is_set() and self._watched:
            jobs_to_remove = []

            for job_id, job in list(self._watched.items()):
                cluster = self._clusters.get(job.cluster_name)
                if not cluster:
                    logger.warning(f"No cluster '{job.cluster_name}' for job {job_id}")
                    continue

                # Check job status
                try:
                    code, output, _ = cluster.ssh.execute_command(
                        f"squeue -j {shlex.quote(job.job_id)} -h -o '%T %N'"
                    )
                    if code != 0:
                        continue  # Transport failure is not a terminal job state.
                    parts = output.strip().split(maxsplit=1)
                    status = parts[0] if parts else None
                    node = parts[1] if len(parts) == 2 else None

                    # Handle empty output (job completed/cancelled)
                    if not status:
                        # Check sacct for final state
                        exit_code, sacct_out, _ = cluster.ssh.execute_command(
                            f"sacct -j {shlex.quote(job.job_id)} -n -o State -X"
                        )
                        if exit_code == 0 and sacct_out.strip():
                            status = sacct_out.strip().split()[0].rstrip("+")
                        else:
                            status = "UNKNOWN"

                    job.status = status
                    if node:
                        job.node = node.strip()

                    logger.debug(f"Job {job_id}: status={status}, node={job.node}")

                    # Check if ready
                    if status == "RUNNING":
                        custom_cond = self._custom_conditions.get(job_id)

                        if custom_cond:
                            # Check custom condition
                            try:
                                if custom_cond(job):
                                    self._trigger_callback(job, job_id)
                                    jobs_to_remove.append(job_id)
                            except Exception as e:
                                logger.error(
                                    f"Custom condition failed for {job_id}: {e}"
                                )
                        else:
                            # No custom condition, just RUNNING is enough
                            self._trigger_callback(job, job_id)
                            jobs_to_remove.append(job_id)

                    elif status in [
                        "COMPLETED",
                        "FAILED",
                        "CANCELLED",
                        "TIMEOUT",
                        "NODE_FAIL",
                    ]:
                        logger.warning(f"Job {job_id} reached terminal state: {status}")
                        jobs_to_remove.append(job_id)

                except Exception as e:
                    logger.error(f"Error checking job {job_id}: {e}")

            # Remove completed jobs
            for job_id in jobs_to_remove:
                self.remove_job(job_id)

            # Calculate sleep interval (minimum across all watched jobs)
            if self._watched:
                intervals = [self.get_poll_interval(j) for j in self._watched.values()]
                sleep_time = min(intervals) if intervals else 60
                logger.debug(f"Sleeping {sleep_time}s (next poll)")

                # The event interrupts a long polling interval immediately.
                self._stop_event.wait(sleep_time)

            self._persist()

        logger.info("Watch loop exited")

    def _trigger_callback(self, job: WatchedJob, key: Optional[str] = None):
        """Trigger the callback for a job."""
        callback = self._callbacks.get(key or job.job_id)
        if callback:
            logger.info(f"Triggering callback for job {job.job_id}")
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Callback failed for job {job.job_id}: {e}")
        else:
            logger.info(f"Job {job.job_id} is ready (no callback registered)")

    def _persist(self):
        """Save watched jobs to disk for recovery."""
        if not self._save_state:
            return
        self.project_dir.mkdir(parents=True, exist_ok=True)

        data = {job_id: job.to_dict() for job_id, job in self._watched.items()}

        # Replace atomically so a stopped process cannot leave a partial JSON file.
        temporary = self.jobs_file.with_suffix(".tmp")
        with open(temporary, "w") as f:
            json.dump(data, f, indent=2)
        temporary.replace(self.jobs_file)

        logger.debug(f"Persisted {len(data)} jobs to {self.jobs_file}")

    def _load(self):
        """Load watched jobs from disk."""
        if not self.jobs_file.exists():
            return

        try:
            with open(self.jobs_file) as f:
                data = json.load(f)

            for job_id, job_data in data.items():
                self._watched[job_id] = WatchedJob.from_dict(job_data)

            logger.info(f"Loaded {len(self._watched)} jobs from {self.jobs_file}")
        except Exception as e:
            logger.warning(f"Failed to load jobs file: {e}")

    @property
    def pending_count(self) -> int:
        """Number of jobs still being watched."""
        return len(self._watched)

    def list_jobs(self) -> Dict[str, WatchedJob]:
        """Get all watched jobs."""
        return dict(self._watched)
