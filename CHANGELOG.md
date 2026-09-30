# Changelog

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
