# ClusterScheduler (`rclust`)

`rclust` sends a Slurm job to whichever of your clusters will start it soonest. Point it at a job
script; it reads the `#SBATCH` lines, asks each cluster you have access to when the job could
start, picks one, copies the script there and runs `sbatch`.

- **Reads your script**: CPUs, GPUs (`--gpus`, `--gpus-per-node`, `--gres=gpu:...`), memory, time,
  nodes, tasks and partition come from the `#SBATCH` header; command-line flags override them.
- **One login per cluster**: connections are opened once with `rclust connect` (answering any
  password or two-factor prompt) and reused through OpenSSH's ControlMaster.
- **Learns from the queue**: `rclust learn` reads how long jobs like yours have been waiting on
  each cluster, and the default policy uses that.
- **Per-cluster accounts**: applies the right `--account` on each cluster, optionally a separate
  one for CPU-only jobs.

## Installation

Needs Python 3.10+, OpenSSH and `rsync` on your machine, and a Linux login node running Slurm on
each cluster.

```bash
uv tool install git+https://github.com/narunraman/rclust
```

(or `pipx install git+...`). For development: clone, `uv sync`, `uv run pytest`.

## Quick start

```bash
rclust config                 # add your clusters (writes ~/.config/rclust/config.yaml)
rclust connect                # log in to each once; answer any two-factor prompts
rclust discover               # optional: list each cluster's GPU types for the config
rclust learn                  # read recent queue history (takes a minute the first time)
rclust suggest job.sh         # where would it go?
rclust submit job.sh          # send it there
```

`test_job.sh` in this repository is a tiny script to try it with.

## Configuration

`rclust` uses the first of these that exists:

1. `--config path/to/config.yaml`
2. `$RCLUST_CONFIG`
3. `./config.yaml` in the current directory
4. `~/.config/rclust/config.yaml` (or `$XDG_CONFIG_HOME/rclust/config.yaml`)

`rclust config` creates and edits it interactively; `config.yaml.example` is a starting point.

```yaml
default_policy: history        # or earliest, balanced
clusters:
  cluster-a:                   # the name you use in rclust commands
    host: hpc1.example.edu     # login host, or an alias from ~/.ssh/config (default: the name)
    user: your-username        # optional: default comes from your SSH config
    account: my-project        # optional: adds --account=my-project
    tags: [gpu]
    gpu_types:                 # optional: GPU names and the Slurm spec to request them with
      - type: h100
        slurm_spec: nvidia_h100_80gb_hbm3
  cluster-b:
    host: hpc2.example.edu
    account: my-project
    cpu_account: my-project-cpu   # optional: account for CPU-only jobs, if your site splits them
    remote_dir: ~/jobs            # where scripts are copied (default ~/cluster_scheduler_jobs)
    resources:
      gpus: ["a100"]              # as printed by `rclust discover`
    # ssh_key: ~/.ssh/id_ed25519  # optional: otherwise your SSH agent / config
```

Anything you can put in `~/.ssh/config` (ports, jump hosts, keys) works: give the alias as `host`.
Clusters without `resources` or `gpu_types` are assumed to be able to run any job; with them, jobs
that need GPUs skip clusters that have none (or not the requested type).

## Usage

```bash
rclust submit job.sh                          # requirements from the script
rclust submit job.sh --gpus 2 --time 12:00:00 # override them
rclust submit job.sh -x cluster-b --wait      # skip a cluster; wait until it's running
rclust submit job.sh --on-running "notify-send started"   # run a local command once it starts
rclust submit job.sh --after-any 123,456      # start after these jobs have ended (afterany)
rclust submit job.sh --remote-dir ~/experiments/run1
rclust suggest --gpus 1 --time 3:00:00 --explain
```

If the script sets neither CPUs nor GPUs, or no time limit, rclust asks for 1 CPU and 1 hour.
Only the script is copied; code, data and environments must already be on each cluster it may go
to. The job runs from your home directory on the cluster, as with a plain `sbatch`. Job IDs in
`--after-any` / `--after-all` belong to one cluster, so combine them with `-x` or `--tags` to keep
the job on that cluster. A directory containing a `submit.run` script (for example a META-Farm
farm) can also be submitted; it is copied to `~/farms/` and run.

Only `connect` (and `submit` and `discover`, which connect first) may show login prompts; other
commands fail fast if a cluster isn't connected, and say to run `rclust connect`. Expected failures (no config, no suitable cluster, a failed submission) print a
one-line error and exit with status 1; usage errors exit with 2.

### Scheduling policies

- **history** (default): if `sbatch --test-only` says some cluster can start the job within 5
  minutes, send it there; otherwise send it where jobs of the same shape had the lowest average
  wait (each wait capped at 24 hours) over the past week, from `rclust learn`. Until you have run
  `rclust learn` it behaves like `earliest`.
- **earliest**: asks each cluster when the job would start (`sbatch --test-only`) and picks the
  soonest. Near-ties (within 5 minutes) go to the cluster where your fairshare is higher.
- **balanced**: favours clusters where your fairshare is high and you have few jobs running,
  skipping any with a wait over 24 hours.

Old names still work: `learned` means `history`; `rush` and `queue-time` mean `earliest`.
Start-time estimates are converted from each cluster's time zone before they are compared.

### Queue history

`rclust learn` reads how long everyone's jobs have waited on each cluster, from Slurm's accounting
records (`sacct -a`). It keeps only each job's shape (GPUs, CPUs, memory, time limit, partition) and
timing, never user or account names, in `~/.local/share/rclust/history.sqlite`. The first read covers
2 days (`--days` for more); after that, `rclust learn` and `rclust connect` fetch only what's new.

```bash
rclust learn
rclust suggest --gpus 1 --time 3:00:00 --explain
```
```
Jobs like yours (1 GPU, ~3:00:00), last 7 days
cluster     jobs  median wait  80% started within  avg wait (≤24h)
cluster-a  2,063           4m                1h08              41m
cluster-b  1,932          29m               10h07            3h12
```

A job's wait runs from when it became eligible (not when it was submitted, so holds and dependencies
don't count) to when it started. Jobs still waiting, or cancelled while waiting, count as "waited at
least this long", and the figures are Kaplan-Meier estimates, so they aren't skewed toward the jobs
that got lucky. Some sites hide other users' jobs from `sacct`; there you'll only see your own.

### Why the history policy is this simple

Before settling on it, we fitted several models of queue waits (a censored log-normal regression, a
two-part "instant start or real wait" mixture, and gradient-boosted hazard models, choosing among
clusters by Thompson sampling) and tested them on three weeks of history from three large national
clusters. What the data showed made most of that unnecessary:

- **The choice matters.** On a typical day the best cluster beats the worst by 1-6 hours for CPU
  jobs and 5-8 hours for multi-GPU jobs of the same shape.
- **But the best cluster changes almost at random from day to day.** Yesterday's best was today's
  best only about half the time, barely above chance, so there's little day-level signal for any
  model to find.
- **Clusters have lasting characters,** so a steady view beats chasing recent winners. Extra wait
  over perfect hindsight, averaged over days:

  | rule | CPU jobs | 1 GPU | 4+ GPUs |
  |---|---|---|---|
  | random cluster | +1.3 h | +1.9 h | +2.9 h |
  | yesterday's best | +0.6 h | +1.2 h | +2.0 h |
  | best of the past 7 days | +0.3 h | +1.4 h | +1.3 h |

- **No exploration is needed.** `sacct -a` shows everyone's jobs, so every cluster is observed
  without sending work to it.

The one live signal that should override the history is Slurm reporting that a job could start
right now (an open backfill gap), which the history can't know about.

## Python API

```python
from rclust import Scheduler, JobSpec

scheduler = Scheduler()                       # same config lookup as the CLI
cluster, metrics = scheduler.select_cluster(JobSpec(gpus=1, time="3:00:00"))
job = scheduler.submit("train.sh")            # parse, pick a cluster, submit; returns a JobHandle
scheduler.wait_until(job)                     # block until it is running
```

`Scheduler(ssh_provider=...)` takes a function from cluster name to an SSH client with an
`execute_command(cmd, timeout=..., use_login_shell=...)` method, for applications that manage their
own connections. The package was called `scheduler` before 0.2; `import scheduler` still works but
warns.
