# rclust

rclust sends a Slurm job to whichever of your clusters will start it soonest. Point it at a job script: it reads
the `#SBATCH` lines, works out where jobs of that shape start fastest, copies the script there and runs `sbatch`
with the right account. You log in to each cluster once, answering any two-factor prompt, and every later command
reuses the connection. It is also the cluster layer of
<a href="https://github.com/narunraman/slurm-llm#gh-light-mode-only"><img src=".github/assets/github_logo_light.svg" height="14" alt=""></a><a href="https://github.com/narunraman/slurm-llm#gh-dark-mode-only"><img src=".github/assets/github_logo_dark.svg" height="14" alt=""></a> [slurm-llm](https://github.com/narunraman/slurm-llm).

## Run it

You need Python 3.10+, OpenSSH, `rsync` and [uv](https://docs.astral.sh/uv/) on your machine, and Slurm on each
cluster's login node.

```bash
uv tool install git+https://github.com/narunraman/rclust@v0.4.0
```

```bash
rclust config                 # add your clusters (writes ~/.config/rclust/config.yaml)
rclust connect                # log in to each once
rclust discover --save        # optional: record each cluster's GPU types
rclust learn                  # read the past week's queue history (a minute the first time)
rclust suggest job.sh         # where would it go, and why
rclust submit job.sh          # send it there
```

[`test_job.sh`](test_job.sh) is a tiny script to try it with. Flags override the script's `#SBATCH` lines:

```bash
rclust submit job.sh --gpus h100:2 --time 12:00:00
rclust submit job.sh -x cluster-b --wait               # skip a cluster; wait until it is running
rclust submit job.sh --after-any 123,456 -x cluster-b  # dependencies are per cluster, so pin it to one
```

Only the script is copied: code, data and environments must already be on each cluster it may go to.

## Commands

| Command | What it does |
|---|---|
| `rclust config [CLUSTER]` | Create or edit the config (`--add` a cluster, `--example` prints an annotated one) |
| `rclust connect [CLUSTER...]` | Log in once and keep the connection open (`-p 4h` to limit it) |
| `rclust discover [CLUSTER...]` | List each cluster's GPU types and their memory (`--save` writes them to the config) |
| `rclust learn` | Read queue history from Slurm's accounting records |
| `rclust suggest [SCRIPT]` | Say which cluster a job would go to (`--explain` shows the waits) |
| `rclust submit SCRIPT` | Pick a cluster and submit (`--wait`, `--on-running CMD`, `--after-any`/`--after-all`) |

Only `connect`, `discover` and `submit` may show a login prompt; the others fail at once on a cluster that is not
connected.

## Configuration

rclust reads `--config PATH`, else `$RCLUST_CONFIG`, else `./config.yaml`, else `~/.config/rclust/config.yaml`.

```yaml
default_policy: history          # or earliest, balanced
clusters:
  cluster-a:                     # the name you use in commands
    host: hpc1.example.edu       # login host, or an alias from ~/.ssh/config
    account: my-project          # optional: --account
    cpu_account: my-project-cpu  # optional: for CPU-only jobs, if your site splits them
    gpu_types:
      - type: h100
        slurm_spec: nvidia_h100_80gb_hbm3   # the name Slurm expects in --gpus
```

Anything in `~/.ssh/config` (ports, jump hosts, keys) works if you give its alias as `host`. A cluster without
`gpu_types` is assumed to run any job; with them, GPU jobs skip clusters that lack the requested type.

## How it chooses

The default policy, **history**, sends a job to a cluster where `sbatch --test-only` says it can start within 5
minutes; otherwise to the cluster where jobs of the same shape waited least over the past week. `rclust learn`
reads those waits from `sacct -a`, keeping only each job's shape and timing (never user or account names), in
`~/.local/share/rclust/history.sqlite`:

```
$ rclust suggest --gpus 1 --time 3:00:00 --explain
Jobs like yours (1 GPU, ~3:00:00), last 7 days
cluster     jobs  median wait  80% started within  avg wait (≤24h)
cluster-a  2,063           4m                1h08              41m
cluster-b  1,932          29m               10h07            3h12
```

**earliest** asks every cluster for its start estimate and takes the soonest; **balanced** favours clusters where
your fairshare is high and you have few jobs running.

We tried fitted models of queue waits (censored regressions, mixtures, boosted hazard models with Thompson sampling)
on three weeks of history from three national clusters. The best cluster changes almost at random from day to day,
while each cluster's typical wait is stable, so the past week's average does as well as the models. Extra wait over
perfect hindsight:

| rule | CPU jobs | 1 GPU | 4+ GPUs |
|---|---|---|---|
| random cluster | +1.3 h | +1.9 h | +2.9 h |
| yesterday's best | +0.6 h | +1.2 h | +2.0 h |
| best of the past 7 days | +0.3 h | +1.4 h | +1.3 h |

## Python

```python
from rclust import Scheduler, JobSpec

scheduler = Scheduler()                       # same config lookup as the CLI
cluster, metrics = scheduler.select_cluster(JobSpec(gpus=1, time="3:00:00"))
job = scheduler.submit("train.sh")
scheduler.wait_until(job)                     # block until it is running
```

The package installs as `clusterscheduler` and imports as `rclust`.

## Tests

```bash
uv run pytest                                                        # offline; never touches a cluster
RCLUST_TEST_CLUSTERS=cluster-a uv run pytest tests/integration -v    # read-only checks on your clusters
RCLUST_TEST_CLUSTERS=cluster-a RCLUST_TEST_SUBMIT=1 uv run pytest tests/integration -v   # also submits one 1-minute job
```

The integration tests reuse connections from `rclust connect` and skip clusters that are not connected.

## License

[MIT](LICENSE).
