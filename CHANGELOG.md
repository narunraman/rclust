# Changelog

## 0.3.3

- `rclust discover` offers to save the GPU types it finds into the config (`--save` to save without
  asking, `--no-save` to only print). Saving replaces each cluster's `resources: gpus:` list and leaves
  the rest of its entry alone.

## 0.3.2

- Placeholder cluster names in tests (`cluster-a`, `hpc1.example.edu`); no real site names.
- Opt-in real-cluster tests in `tests/integration/`, run against the clusters in your own config:
  `RCLUST_TEST_CLUSTERS=all uv run pytest tests/integration -v` for read-only checks (sinfo,
  squeue, sshare, discover, sacct history, `sbatch --test-only` start estimates and their time-zone
  conversion, GPU-type probes), and `RCLUST_TEST_SUBMIT=1` to also submit one tiny job to each.
  They only reuse connections from `rclust connect`, and are skipped by a plain `uv run pytest`.

## 0.3.1

Fixes from a first run against real clusters.

- **`submit` works with rsync 3.2.4 and later.** Uploads to `~/...` paths (including the default
  `~/cluster_scheduler_jobs` and `~/farms/`) failed because newer rsync no longer lets the remote
  shell expand `$HOME`. Destinations are now relative to the remote home. A failed upload shows
  rsync's error, and removes the directory it created if that is still empty.
- **Skipped clusters are always reported**, with a one-line reason (colour codes and decorative
  banner lines removed), e.g. `skipped cluster-b: sbatch: error: ...`, without needing `-v`.
- **`--gpu-type TYPE`** for `suggest` and `submit`; `--gpus` also accepts `TYPE:N` (e.g. `h100:2`).
- **The choice is explained**: `suggest` and `submit` say which policy picked the cluster and why.
- `suggest --explain` says how much history the table actually covers, and names the GPU type.
- `discover` prints plain YAML (no box borders) on stdout, reads partitions hidden from a plain
  `sinfo`, and parses GRES strings (socket annotations, MIG slices) more robustly.
- An unknown cluster name in `-x/--exclude` is an error rather than silently ignored.
- The example config ships with the package: `rclust config --example`.
- `rclust connect` no longer overrides `ControlPersist` from your `~/.ssh/config`; `-p` still sets it.
- `scripts/publish.sh` creates annotated release tags, so `git push --follow-tags` pushes them.
