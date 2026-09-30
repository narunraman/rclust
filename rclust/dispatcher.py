from typing import List, Optional, Dict, Type
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
import logging
from .config import GlobalConfig
from .cluster import Cluster, SlurmCluster, JobSpec, ClusterMetrics

logger = logging.getLogger(__name__)


class Policy:
    reason: Optional[str] = None  # after select(): why that cluster, in one line

    def select(self, candidates: Dict[Cluster, ClusterMetrics]) -> Optional[Cluster]:
        raise NotImplementedError


class FastestStartPolicy(Policy):
    def select(self, candidates: Dict[Cluster, ClusterMetrics]) -> Optional[Cluster]:
        if not candidates:
            return None

        sorted_clusters = sorted(
            candidates.items(), key=lambda item: item[1].estimated_start_time
        )

        best_cluster, best_metrics = sorted_clusters[0]

        # near-ties (within 5 minutes of the soonest) go to the highest fairshare
        near_ties = []
        for cluster, metrics in sorted_clusters:
            diff = metrics.estimated_start_time - best_metrics.estimated_start_time
            if abs(diff.total_seconds()) < 300:  # 5 minutes
                near_ties.append((cluster, metrics))
            else:
                break

        if len(near_ties) > 1:
            near_ties.sort(key=lambda item: item[1].fairshare, reverse=True)
            self.reason = "soonest estimated start (a near-tie, given to your higher fairshare)"
            return near_ties[0][0]

        self.reason = "soonest estimated start"
        return best_cluster


class BalancedPolicy(Policy):
    def select(self, candidates: Dict[Cluster, ClusterMetrics]) -> Optional[Cluster]:
        if not candidates:
            return None

        # skip clusters with almost no fairshare left or a wait over a day
        filtered_candidates = []
        for cluster, metrics in candidates.items():
            if metrics.fairshare < 0.01:
                logger.debug(f"Skipping {cluster.name}: Low FS ({metrics.fairshare})")
                continue

            wait_duration = metrics.estimated_start_time - datetime.now()
            if wait_duration.total_seconds() > 86400:
                logger.debug(f"Skipping {cluster.name}: Wait > 24h")
                continue

            filtered_candidates.append((cluster, metrics))

        if not filtered_candidates:
            logger.info("balanced: every cluster was filtered out; considering them all")
            filtered_candidates = list(candidates.items())

        # score = fairshare, discounted slightly by how many jobs you already have running
        best_cluster = None
        best_score = -float("inf")

        for cluster, metrics in filtered_candidates:
            load_factor = 1.0 / (1.0 + (metrics.load / 1000.0))
            score = metrics.fairshare * load_factor

            logger.debug(
                f"{cluster.name}: Score={score:.4f} (FS={metrics.fairshare}, Load={metrics.load})"
            )

            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None:
            m = candidates[best_cluster]
            self.reason = (f"highest fairshare ({m.fairshare:.2f}) with few of your jobs running "
                           f"({m.load})")
        return best_cluster


class QueueTimePolicy(Policy):
    """Select cluster with shortest estimated queue time. No other factors considered."""

    def select(self, candidates: Dict[Cluster, ClusterMetrics]) -> Optional[Cluster]:
        if not candidates:
            return None

        self.reason = "soonest estimated start"
        return min(candidates.items(), key=lambda item: item[1].estimated_start_time)[0]


class HistoryPolicy(Policy):
    """Where jobs like this one have started soonest lately, unless one can start it right now.

    Clusters are compared on the average wait (each capped at 24 h) of jobs of the same shape over
    the past week, from `rclust learn`. On three weeks of history from three clusters, the best
    cluster for a kind of job changed almost at random from day to day, but each cluster kept a
    lasting character, so a steady 7-day view beat chasing yesterday's winner. The one live signal
    that overrides it: if `sbatch --test-only` says a cluster can start the job within a few
    minutes, go there. The history covers everyone's jobs, so no exploration is needed.
    """

    START_NOW_S = 300
    DAYS = 7

    def __init__(self, spec: JobSpec | dict[str, JobSpec], history=None):
        self.spec = spec
        self.history = history
        self.summaries: Dict[str, object] = {}  # cluster name -> history.Summary, from scores()

    def scores(self, names: List[str]) -> Dict[str, float]:
        """Average recent wait (seconds, capped at 24 h) for jobs shaped like this, per cluster."""
        from .history import History, history_path, summarize

        history = self.history
        if history is None:
            if not history_path().exists():
                return {}
            history = History()
        try:
            summaries = []
            for name in names:
                spec = self.spec[name] if isinstance(self.spec, dict) else self.spec
                summaries.extend(summarize(history, [name], spec, days=self.DAYS))
            self.summaries = {s.cluster: s for s in summaries}
            return {s.cluster: s.mean24 for s in summaries if s.jobs and s.mean24 is not None}
        finally:
            if self.history is None:
                history.close()

    def select(self, candidates: Dict[Cluster, ClusterMetrics]) -> Optional[Cluster]:
        if not candidates:
            return None
        now = datetime.now()
        ready = [c for c, m in candidates.items()
                 if m.estimated_start_time is not None
                 and (m.estimated_start_time - now).total_seconds() <= self.START_NOW_S]
        pool = ready or list(candidates)
        scores = self.scores([c.name for c in pool])
        known = [c for c in pool if c.name in scores]
        if known:
            best = min(known, key=lambda c: scores[c.name])
            self.reason = self._explain(best, ready)
            return best
        # no history for these clusters: fall back to Slurm's estimates
        best = FastestStartPolicy().select({c: candidates[c] for c in pool})
        if ready:
            self.reason = f"{best.name} can start it now"
        else:
            self.reason = ("soonest estimated start (no queue history for these clusters yet; "
                           "run `rclust learn`)")
        return best

    def _explain(self, best: Cluster, ready: List[Cluster]) -> str:
        from .history import fmt_span, fmt_wait

        if ready:
            if len(ready) == 1:
                return f"{best.name} can start it now"
            return (f"{best.name} can start it now, and had the shortest recent waits "
                    f"of the {len(ready)} that can")
        summary = self.summaries.get(best.name)
        span = getattr(summary, "span_s", None)
        period = ("the past week" if span is None or span >= (self.DAYS - 0.5) * 86400
                  else f"the past {fmt_span(span, self.DAYS)}")
        like = "jobs with this many GPUs" if getattr(summary, "loose", False) else "jobs like this"
        detail = f" (median {fmt_wait(summary.median)})" if summary is not None else ""
        return f"{like} waited least on {best.name} over {period}{detail}"


class Dispatcher:
    # history: where jobs like this started soonest over the past week, unless a cluster can start
    #          it right now (see HistoryPolicy); behaves like earliest until `rclust learn` has run
    # earliest: soonest estimated start (sbatch --test-only); near-ties go to the higher fairshare
    # balanced: favour clusters where your fairshare is high and the load is low
    POLICIES: Dict[str, Type[Policy]] = {
        "history": HistoryPolicy,
        "earliest": FastestStartPolicy,
        "balanced": BalancedPolicy,
    }
    # old names, still accepted
    ALIASES: Dict[str, str] = {"rush": "earliest", "queue-time": "earliest", "learned": "history"}

    def __init__(self, config: GlobalConfig, ssh_provider=None):
        """
        Initialize dispatcher.

        Args:
            config: Global configuration
            ssh_provider: Optional function(cluster_name) -> SSH client.
                         If provided, clusters use external SSH connections.
        """
        self.config = config
        self.ssh_provider = ssh_provider
        self.clusters = [
            SlurmCluster(c, ssh_provider=ssh_provider) for c in config.clusters
        ]
        self.last_metrics: Dict[Cluster, ClusterMetrics] = {}  # from the last propose_cluster
        self.last_errors: Dict[str, str] = {}   # cluster name -> why its probe failed
        self.last_problem: Optional[str] = None  # why the last propose_cluster found nothing
        self.last_reason: Optional[str] = None   # why it chose what it chose: "policy: reason"

    def _fail(self, problem: str) -> None:
        self.last_problem = problem
        logger.info(problem)
        return None

    def get_cluster_by_name(self, name: str) -> Optional[Cluster]:
        for c in self.clusters:
            if c.name == name:
                return c
        return None

    def check_resources(self, cluster: Cluster, job_spec: JobSpec) -> bool:
        """False if the config says the cluster lacks what the job needs (only GPUs are checked).
        Clusters without a `resources` section are assumed to be able to run anything."""
        res = cluster.resources
        if not res:
            return True
        if job_spec.gpus and job_spec.gpus > 0:
            cluster_gpus = res.get("gpus", [])
            if not cluster_gpus:
                return False

            if job_spec.gpu_type:
                supported = set(cluster_gpus)
                supported.update(g.get("type") for g in res.get("gpu_types", []))
                supported.update(g.get("slurm_spec") for g in res.get("gpu_types", []))
                if job_spec.gpu_type not in supported:
                    return False
        return True

    def propose_cluster(
        self,
        job_spec: JobSpec | dict[str, JobSpec],
        policy_name: Optional[str] = None,
        required_tags: Optional[List[str]] = None,
        exclude_clusters: Optional[List[str]] = None,
    ) -> Optional[Cluster]:
        """
        Propose the best cluster for a job.

        Args:
            job_spec: Either a single JobSpec (used for all clusters) or a dict
                     mapping cluster_name -> JobSpec for per-cluster requirements.
                     When dict is provided, clusters not in the dict are excluded.
            policy_name: Selection policy ("earliest" or "balanced"); None uses the config default
            required_tags: Tags that clusters must have
            exclude_clusters: Cluster names to exclude

        Returns:
            Selected Cluster or None if no suitable cluster found
        """
        per_cluster_specs: dict[str, JobSpec] = job_spec if isinstance(job_spec, dict) else {}
        default_spec: JobSpec | None = None if isinstance(job_spec, dict) else job_spec
        self.last_metrics = {}
        self.last_errors = {}
        self.last_problem = None
        self.last_reason = None

        def spec_for(cluster: Cluster) -> JobSpec:
            if per_cluster_specs:
                return per_cluster_specs[cluster.name]
            return default_spec or JobSpec(cpus=1, time="1:00:00")

        candidates = [c for c in self.clusters if c.name not in set(exclude_clusters or ())]
        if isinstance(job_spec, dict):
            candidates = [c for c in candidates if c.name in per_cluster_specs]
        if not candidates:
            return self._fail("no clusters left after exclusions")
        if required_tags:
            candidates = [c for c in candidates if set(required_tags).issubset(c.tags)]
            if not candidates:
                return self._fail(f"no cluster has all the tags {', '.join(required_tags)}")
        candidates = [c for c in candidates if self.check_resources(c, spec_for(c))]
        if not candidates:
            return self._fail("no cluster has the GPUs this job asks for (see resources in the config)")

        name = policy_name or self.config.default_policy
        name = self.ALIASES.get(name, name)
        if name not in self.POLICIES:
            raise ValueError(f"unknown policy {name!r}; choose from {', '.join(self.POLICIES)}")

        def probe(cluster):
            try:
                metrics = cluster.get_metrics(spec_for(cluster))
                return metrics, None if metrics else (getattr(cluster, "last_error", None) or "no estimate")
            except Exception as e:
                return None, str(e) or type(e).__name__

        cluster_metrics: Dict[Cluster, ClusterMetrics] = {}
        with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as executor:
            futures = {executor.submit(probe, c): c for c in candidates}
            try:
                for future in as_completed(futures, timeout=120):
                    cluster = futures[future]
                    metrics, error = future.result()
                    if metrics:
                        cluster_metrics[cluster] = metrics
                        logger.info(f"{cluster.name}: estimated start {metrics.estimated_start_time}")
                    else:
                        self.last_errors[cluster.name] = error
                        logger.info(f"{cluster.name}: probe failed: {error}")
            except FuturesTimeout:
                for future, cluster in futures.items():
                    if not future.done():
                        self.last_errors[cluster.name] = "probe timed out"

        self.last_metrics = cluster_metrics
        if not cluster_metrics:
            return self._fail("could not get a start estimate from any cluster")

        # 3. Apply Policy
        if name == "history":
            policy = HistoryPolicy(per_cluster_specs or default_spec or JobSpec())
        else:
            policy = self.POLICIES[name]()
        selected = policy.select(cluster_metrics)

        if selected:
            reason = getattr(policy, "reason", None)
            self.last_reason = f"{name}: {reason}" if reason else name
            logger.info(f"Dispatcher selected: {selected.name} ({self.last_reason})")

        return selected
