"""The `rclust` command line. Heavy modules are imported inside commands so --help stays fast."""

import subprocess
import sys

import rich_click as click

click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.ERRORS_SUGGESTION = "Try 'rclust --help' or 'rclust COMMAND --help'."

POLICIES = ("history", "earliest", "balanced")
POLICY_ALIASES = {"learned": "history", "rush": "earliest", "queue-time": "earliest"}  # old names

_console = None


def console():
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console()
    return _console


def fail(message: str, hint: str = None):
    """Report an expected failure (no traceback) and exit with status 1."""
    from rich.markup import escape

    console().print(f"[bold red]Error:[/bold red] {escape(message)}")
    if hint:
        console().print(f"[dim]{escape(hint)}[/dim]")
    sys.exit(1)


def _esc(text):
    from rich.markup import escape

    return escape(str(text))


class Context:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.scheduler = None


@click.group()
@click.option("--config", default=None, metavar="PATH", envvar="RCLUST_CONFIG",
              help="Config file. Default: $RCLUST_CONFIG, else ./config.yaml, "
                   "else ~/.config/rclust/config.yaml.")
@click.option("-v", "--verbose", count=True, help="Log more (-v: progress, -vv: debug).")
@click.version_option(package_name="clusterscheduler", prog_name="rclust")
@click.pass_context
def main(ctx, config, verbose):
    """
    Send a Slurm job to whichever of your clusters will start it soonest.

    Typical first run: [bold]rclust config[/bold] (add clusters), [bold]rclust connect[/bold]
    (log in once), [bold]rclust learn[/bold] (read queue history), then
    [bold]rclust suggest job.sh[/bold] or [bold]rclust submit job.sh[/bold].
    """
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if verbose:
        logging.getLogger("rclust").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)
    ctx.obj = Context(config)


def get_scheduler(ctx):
    """The Scheduler for this invocation, or a clean exit if the config is missing or invalid."""
    if ctx.obj.scheduler is None:
        from .api import Scheduler

        try:
            ctx.obj.scheduler = Scheduler(ctx.obj.config_path)
        except (OSError, ValueError) as e:
            fail(str(e))
    return ctx.obj.scheduler


def _connect_all(clusters, persist=None):
    """Open (or reuse) a connection to each cluster. Returns the names that failed."""
    failed = set()
    for cluster in clusters:
        try:
            cluster.ssh.connect(persist=persist) if persist else cluster.ssh.connect()
        except Exception as e:
            console().print(f"[red]✗ {cluster.name}:[/red] {_esc(e)}")
            failed.add(cluster.name)
    return failed


def _report_no_cluster(scheduler):
    dispatcher = scheduler.dispatcher
    lines = [f"  {name}: {error}" for name, error in sorted(dispatcher.last_errors.items())]
    message = "no suitable cluster found"
    if dispatcher.last_problem:
        message += f": {dispatcher.last_problem}"
    if lines:
        message += "\n" + "\n".join(lines)
    hint = None
    if any("not connected" in e for e in dispatcher.last_errors.values()):
        hint = "Open connections first with `rclust connect`."
    fail(message, hint)


def _format_wait(estimated_start):
    from datetime import datetime, timedelta

    seconds = int(max(0, (estimated_start - datetime.now()).total_seconds()))
    return str(timedelta(seconds=seconds))


# --- config ---------------------------------------------------------------------------------------

@main.command()
@click.argument("cluster_name", required=False)
@click.option("--add", is_flag=True, help="Add a new cluster.")
@click.pass_context
def config(ctx, cluster_name, add):
    """
    Create or edit the config file interactively.

    Edits the file rclust would load (see --config), or creates
    ~/.config/rclust/config.yaml if there is none yet.

    \b
    Examples:
      rclust config              # list clusters; add or edit one
      rclust config cluster-a    # edit cluster-a
      rclust config --add        # add a cluster
    """
    import yaml
    from rich.prompt import Confirm, Prompt
    from .config import find_config, user_config_path

    path = find_config(ctx.obj.config_path) or user_config_path()
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (yaml.YAMLError, OSError) as e:
            fail(f"could not read {path}: {e}")
        if not isinstance(data, dict) or not isinstance(data.get("clusters") or {}, dict):
            fail(f"{path}: expected a mapping with a 'clusters' mapping inside")
    else:
        data = {}
    clusters = data.setdefault("clusters", {}) or {}
    data["clusters"] = clusters
    out = console()

    if not add and not cluster_name:
        out.print(f"\n[bold]Clusters in {path}:[/bold]")
        names = list(clusters)
        if not names:
            out.print("  (none yet)")
        for i, name in enumerate(names, 1):
            out.print(f"  {i}. {name} [dim]({(clusters[name] or {}).get('host') or name})[/dim]")
        choice = Prompt.ask("\nNumber to edit, [green]A[/green] to add, [red]Q[/red] to quit", default="A")
        if choice.upper() == "Q":
            return
        if choice.upper() != "A":
            if not (choice.isdigit() and 1 <= int(choice) <= len(names)):
                fail(f"invalid choice: {choice}")
            cluster_name = names[int(choice) - 1]

    if cluster_name and cluster_name in clusters:
        entry = clusters[cluster_name] or {}
        out.print(f"\n[bold cyan]Editing {cluster_name}[/bold cyan]")
    else:
        if cluster_name and not add and not Confirm.ask(f"No cluster '{cluster_name}'. Add it?"):
            return
        out.print("\n[bold cyan]New cluster[/bold cyan]")
        cluster_name = cluster_name or Prompt.ask("Name (used in rclust commands, e.g. cluster-a)")
        if cluster_name in clusters:
            fail(f"cluster '{cluster_name}' already exists; edit it with `rclust config {cluster_name}`")
        entry = {}

    entry["host"] = Prompt.ask("Login host or SSH alias", default=entry.get("host") or cluster_name)
    user = Prompt.ask("Username (blank: use your SSH config)", default=entry.get("user") or "")
    account = Prompt.ask("Slurm account (optional)", default=entry.get("account") or "")
    tags = Prompt.ask("Tags, comma separated (optional, e.g. gpu)", default=", ".join(entry.get("tags") or []))
    for key, value in (("user", user), ("account", account)):
        if value:
            entry[key] = value
        else:
            entry.pop(key, None)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if tag_list:
        entry["tags"] = tag_list
    else:
        entry.pop("tags", None)

    clusters[cluster_name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_config(path, data)
    out.print(f"\n[green]✓ Saved {cluster_name} to {path}[/green]")
    out.print("[dim]Next: `rclust connect` to log in, `rclust discover` to list its GPU types, "
              "`rclust learn` to read queue history.[/dim]")


def _save_config(path, data):
    """Write the config atomically, so an interrupted write cannot truncate it."""
    import os
    import tempfile
    from pathlib import Path

    import yaml

    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".rclust-config-")
    try:
        with os.fdopen(descriptor, "w") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


# --- connect / discover ---------------------------------------------------------------------------

@main.command()
@click.argument("clusters", nargs=-1)
@click.option("--persist", "-p", default=None, metavar="DURATION",
              help="How long an idle connection stays open (ssh ControlPersist, e.g. 30m, 4h). Default: 1h.")
@click.pass_context
def connect(ctx, clusters, persist):
    """
    Log in to clusters once and keep the connections open.

    Password and two-factor prompts happen here; later commands reuse the
    connection (ssh ControlMaster). If you have run `rclust learn` before,
    this also fetches new queue history.

    \b
    Examples:
      rclust connect                  # all clusters
      rclust connect cluster-a        # just this one
      rclust connect -p 4h            # keep idle connections for 4 hours
    """
    scheduler = get_scheduler(ctx)
    targets = _select(scheduler, clusters)
    failed = _connect_all(targets, persist)
    ok = [c for c in targets if c.name not in failed]
    for cluster in ok:
        console().print(f"[green]✓ {cluster.name}[/green]")

    from . import history as hist

    if ok and hist.history_path().exists():
        _learn(ok, days=None)
    if failed:
        fail(f"could not connect to {', '.join(sorted(failed))}")


def _select(scheduler, names):
    """Configured clusters with these names (all if none given); unknown names are an error."""
    clusters = scheduler.dispatcher.clusters
    unknown = sorted(set(names) - {c.name for c in clusters})
    if unknown:
        fail(f"unknown cluster(s): {', '.join(unknown)}",
             f"Configured: {', '.join(c.name for c in clusters)}")
    return [c for c in clusters if not names or c.name in names]


@main.command()
@click.argument("clusters", nargs=-1)
@click.pass_context
def discover(ctx, clusters):
    """
    List the GPU types on each cluster.

    Prints a `resources` block per cluster to paste into its entry in the
    config, so jobs that need GPUs skip clusters without them.
    """
    import re

    scheduler = get_scheduler(ctx)
    targets = _select(scheduler, clusters)
    failed = _connect_all(targets)
    out = console()
    lines = []
    for cluster in targets:
        if cluster.name in failed:
            continue
        try:
            with out.status(f"Probing {cluster.name}..."):
                gpus = cluster.discover_resources().get("gpus", [])
        except Exception as e:
            out.print(f"[red]✗ {cluster.name}:[/red] {_esc(e)}")
            failed.add(cluster.name)
            continue
        lines.append(f"  {cluster.name}:")
        if gpus:
            lines += ["    resources:", "      gpus:"]
            lines += [f'        - "{g}"' + ("  # MIG slice" if re.search(r"\d+g\.\d+gb", g) else "")
                      for g in gpus]
        else:
            lines.append("    # no GPUs found")
    if lines:
        from rich.panel import Panel

        out.print(Panel("clusters:\n" + "\n".join(lines), title="Add to each cluster in your config",
                        border_style="green"))
    if failed:
        fail(f"could not probe {', '.join(sorted(failed))}")


# --- suggest / submit -----------------------------------------------------------------------------

def _policy_callback(ctx, param, value):
    if value is None:
        return None
    value = POLICY_ALIASES.get(value, value)
    if value not in POLICIES:
        raise click.BadParameter(f"choose from {', '.join(POLICIES)}")
    return value


def resource_options(f):
    for option in reversed([
        click.option("--cpus", type=click.IntRange(min=1), help="CPUs per task."),
        click.option("--gpus", type=click.IntRange(min=0), help="GPUs."),
        click.option("--mem", metavar="SIZE", help="Memory per node, e.g. 16G."),
        click.option("--time", metavar="TIME", help="Time limit, e.g. 2:00:00."),
        click.option("--policy", metavar="[history|earliest|balanced]", callback=_policy_callback,
                     help="history: where jobs like this waited least over the past week (from "
                          "`rclust learn`), unless a cluster can start it within 5 minutes. "
                          "earliest: soonest start by Slurm's estimate. balanced: high fairshare, "
                          "low load. Default: default_policy in the config, else history."),
        click.option("--tags", multiple=True, metavar="TAG", help="Only clusters with this tag (repeatable)."),
        click.option("--exclude", "-x", multiple=True, metavar="CLUSTER", help="Skip this cluster (repeatable)."),
    ]):
        f = option(f)
    return f


def _job_spec(scheduler, script, cpus, gpus, mem, time):
    """Requirements from the script's #SBATCH header (if any), overridden by command-line flags."""
    from .cluster import JobSpec

    spec = JobSpec()
    if script:
        try:
            spec = scheduler.analyze(script)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            fail(f"could not read {script}: {e}")
    for field, value in (("cpus", cpus), ("gpus", gpus), ("memory", mem), ("time", time)):
        if value is not None:
            setattr(spec, field, value)
    if mem is not None:
        spec.memory_per_cpu = None
    if gpus == 0:
        spec.gpu_type = None
    return spec


def _with_probe_defaults(spec):
    try:
        spec.to_sbatch_args()
    except ValueError as e:
        fail(f"invalid job requirements: {e}")
    if spec.cpus is None and not spec.gpus:
        spec.cpus = 1
    if not spec.time:
        spec.time = "1:00:00"
    return spec


@main.command()
@click.argument("script", required=False, type=click.Path(exists=True, dir_okay=False))
@resource_options
@click.option("--explain", is_flag=True, help="Also show how long jobs like this waited on each cluster.")
@click.option("--days", type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True,
              help="With --explain: how many days of history to use.")
@click.pass_context
def suggest(ctx, script, cpus, gpus, mem, time, policy, tags, exclude, explain, days):
    """
    Show which cluster a job would go to, without submitting it.

    Reads requirements from SCRIPT's #SBATCH lines if given; flags override
    them. With neither, probes for 1 CPU for 1 hour.

    \b
    Examples:
      rclust suggest job.sh
      rclust suggest --gpus 1 --time 3:00:00 --explain
    """
    scheduler = get_scheduler(ctx)
    spec = _job_spec(scheduler, script, cpus, gpus, mem, time)
    if explain:
        _explain(scheduler, spec, days, exclude)
    spec = _with_probe_defaults(spec)
    with console().status("Asking each cluster when the job could start..."):
        cluster, metrics = scheduler.select_cluster(spec, policy=policy, tags=list(tags),
                                                    exclude=list(exclude))
    if not cluster:
        _report_no_cluster(scheduler)
    from rich.panel import Panel

    estimate = (f"Slurm's start estimate there: {_format_wait(metrics.estimated_start_time)} from now"
                if metrics else "Slurm gave no start estimate")
    console().print(Panel(f"[bold green]{cluster.name}[/bold green]\n[dim]{estimate}[/dim]",
                          title="Suggested cluster", border_style="green", expand=False))


@main.command()
@click.argument("target", type=click.Path(exists=True))
@resource_options
@click.option("--remote-dir", metavar="DIR",
              help="Where to copy the script on the cluster. Default: remote_dir in the config, "
                   "else ~/cluster_scheduler_jobs.")
@click.option("--wait", is_flag=True, help="Wait until the job is running.")
@click.option("--on-running", metavar="COMMAND",
              help="Wait until the job is running, then run COMMAND locally.")
@click.option("--after-any", metavar="IDS",
              help="Start once these jobs (comma-separated IDs) have ended, in any state (afterany).")
@click.option("--after-all", metavar="IDS",
              help="Start once these jobs (comma-separated IDs) have all completed successfully (afterok).")
@click.pass_context
def submit(ctx, target, remote_dir, cpus, gpus, mem, time, policy, tags, exclude, wait, on_running,
           after_any, after_all):
    """
    Submit a job script to the best cluster.

    Reads requirements from the script's #SBATCH lines (flags override them),
    asks each cluster when the job could start, picks one by the policy,
    copies the script there and runs sbatch. TARGET may also be a directory
    with a ./submit.run script (e.g. a META-Farm farm), which is copied to
    ~/farms/ and run.

    \b
    Examples:
      rclust submit job.sh
      rclust submit job.sh --gpus 2 --time 12:00:00
      rclust submit job.sh -x cluster-a --wait
      rclust submit job.sh --on-running "notify-send started"
      rclust submit job.sh --after-any 123,456
    """
    from pathlib import Path

    is_dir = Path(target).is_dir()
    if after_any and after_all:
        raise click.UsageError("use only one of --after-any and --after-all")
    if is_dir and (wait or on_running or after_any or after_all or remote_dir):
        raise click.UsageError("--wait, --on-running, --after-* and --remote-dir need a job script, not a directory")

    scheduler = get_scheduler(ctx)
    spec = _job_spec(scheduler, None if is_dir else target, cpus, gpus, mem, time)
    if after_any or after_all:
        ids = [i.strip() for i in (after_any or after_all).split(",")]
        if not all(i.isdigit() for i in ids):
            raise click.BadParameter("expected comma-separated numeric job IDs",
                                     param_hint="--after-any" if after_any else "--after-all")
        spec.dependency = ("afterany:" if after_any else "afterok:") + ":".join(ids)
    spec = _with_probe_defaults(spec)

    candidates = [c for c in scheduler.dispatcher.clusters
                  if c.name not in exclude and set(tags).issubset(c.tags)
                  and scheduler.dispatcher.check_resources(c, spec)]
    failed = _connect_all(candidates)
    out = console()
    with out.status("Asking each cluster when the job could start..."):
        cluster, metrics = scheduler.select_cluster(spec, policy=policy, tags=list(tags),
                                                    exclude=list(set(exclude) | failed))
    if not cluster:
        _report_no_cluster(scheduler)
    estimate = f", estimated start in {_format_wait(metrics.estimated_start_time)}" if metrics else ""
    out.print(f"Selected [bold]{cluster.name}[/bold]{estimate}")

    try:
        with out.status(f"Submitting to {cluster.name}..."):
            if is_dir:
                out.print(scheduler.submit_farm(cluster, target))
                return
            job = scheduler.submit(target, job_spec=spec, cluster_name=cluster.name, remote_dir=remote_dir)
    except (RuntimeError, ValueError, OSError) as e:
        fail(f"submission to {cluster.name} failed: {e}")
    out.print(f"[green]✓ Submitted job {job.job_id} to {cluster.name}[/green]")

    if on_running:
        _run_when_running(scheduler, job, cluster, on_running)
    elif wait:
        with out.status(f"Waiting for job {job.job_id} to start (Ctrl+C stops waiting, not the job)..."):
            started = scheduler.wait_until(job)
        if not started:
            fail(f"job {job.job_id} did not start before the timeout, or ended without running",
                 f"It may still be queued; check with `squeue -j {job.job_id}` on {cluster.name}.")
        out.print(f"[green]Job {job.job_id} is running[/green]")


def _run_when_running(scheduler, job, cluster, command):
    """Poll until the job runs, then run `command` locally. Blocks: a callback can't outlive rclust."""
    from .watcher import JobWatcher, WatchedJob

    ran, status = [], []

    def callback(watched):
        console().print(f"[green]Job {watched.job_id} is running[/green]; running: {command}")
        ran.append(True)
        status.append(subprocess.run(command, shell=True).returncode)

    watcher = JobWatcher(scheduler.project_dir, clusters={cluster.name: cluster},
                         load_saved=False, persist=False)
    watcher.add_job(WatchedJob(job.job_id, job.cluster_name, job.estimated_start), on_running=callback)
    console().print("Waiting for the job to start (Ctrl+C stops waiting, not the job)...")
    try:
        watcher.start(blocking=True)
    except KeyboardInterrupt:
        watcher.stop()
        raise click.Abort()
    if not ran:
        fail(f"job {job.job_id} ended before it was seen running; the command was not run")
    if status[0]:
        fail(f"the command exited with status {status[0]}")


# --- queue history --------------------------------------------------------------------------------

def _learn(clusters, days=None):
    """Pull new accounting records from each cluster into the local history. Returns failed names."""
    from . import history as hist

    failed = []
    with hist.History() as history:
        for cluster in clusters:
            since = history.last_pull(cluster.name) if days is None else None
            try:
                with console().status(f"Reading {cluster.name}'s queue history..."):
                    jobs, now = hist.fetch(cluster.ssh, since_days=days or 2, since_ts=since)
            except Exception as e:
                console().print(f"[red]✗ {cluster.name}:[/red] {_esc(e)}")
                failed.append(cluster.name)
                continue
            n = history.add(cluster.name, jobs, now)
            console().print(f"[green]✓ {cluster.name}[/green] [dim]{n:,} jobs[/dim]")
    return failed


@main.command()
@click.argument("clusters", nargs=-1)
@click.option("--days", type=click.FloatRange(min=0, min_open=True), default=None,
              help="How far back to read. Default: since the last read, or 2 days the first time.")
@click.pass_context
def learn(ctx, clusters, days):
    """
    Read how long jobs have been waiting on each cluster.

    Pulls Slurm's accounting records (sacct -a) and keeps only each job's
    shape (GPUs, CPUs, memory, time limit, partition) and timing; no user or
    account names. Stored in ~/.local/share/rclust/history.sqlite and used by
    the history policy and `rclust suggest --explain`.

    Busy clusters log 100k+ jobs a day, so the first read can take a minute;
    later reads fetch only what is new. Needs open connections (`rclust connect`).

    \b
    Examples:
      rclust learn                # all clusters
      rclust learn cluster-a      # just this one
      rclust learn --days 7       # read a week back
    """
    scheduler = get_scheduler(ctx)
    failed = _learn(_select(scheduler, clusters), days=days)
    if failed:
        fail(f"could not read history from {', '.join(failed)}")


def _explain(scheduler, spec, days, exclude=()):
    from . import history as hist

    if not hist.history_path().exists():
        console().print("[yellow]No queue history yet; run `rclust learn` first.[/yellow]")
        return
    names = [c.name for c in scheduler.dispatcher.clusters if c.name not in exclude]
    with hist.History() as history:
        if not history.clusters():
            console().print("[yellow]No queue history yet; run `rclust learn` first.[/yellow]")
            return
        summaries = hist.summarize(history, names, spec, days=days)
    from rich.table import Table

    gpus = spec.gpus or 0
    shape = f"{spec.cpus} CPUs" if not gpus and spec.cpus else f"{gpus} GPU{'s' if gpus != 1 else ''}"
    if spec.time:
        shape += f", ~{spec.time}"
    table = Table(title=f"Jobs like yours ({shape}), last {days:g} days", title_justify="left",
                  box=None, pad_edge=False, header_style="dim")
    for column, justify in (("cluster", "left"), ("jobs", "right"), ("median wait", "right"),
                            ("80% started within", "right"), ("avg wait (≤24h)", "right")):
        table.add_column(column, justify=justify)
    best = min((s for s in summaries if s.mean24 is not None and s.jobs), key=lambda s: s.mean24, default=None)
    loose_any = False
    for s in summaries:
        loose_any |= s.loose and s.jobs > 0
        table.add_row(s.cluster + (" ←" if s is best else ""), f"{s.jobs:,}" + ("*" if s.loose and s.jobs else ""),
                      hist.fmt_wait(s.median) if s.jobs else "no data",
                      hist.fmt_wait(s.p80) if s.jobs else "",
                      hist.fmt_wait(int(s.mean24)) if s.jobs and s.mean24 is not None else "")
    console().print(table)
    notes = ["— means too many are still waiting to say.",
             "← is where the history policy would send it, unless a cluster can start it right now."]
    if loose_any:
        notes.insert(0, "* few jobs matched your full shape, so these match on GPU count only.")
    console().print("[dim]" + " ".join(notes) + "[/dim]\n")


if __name__ == "__main__":
    main()
