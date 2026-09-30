"""ONE tiny real job per selected cluster, through rclust's own submit path. Spends a little allocation.

Collected only with RCLUST_TEST_SUBMIT=1 (as well as RCLUST_TEST_CLUSTERS):

    RCLUST_TEST_CLUSTERS=cluster-a RCLUST_TEST_SUBMIT=1 uv run pytest tests/integration -v

The job asks for 1 CPU, 1 minute and 256M, runs `sleep 20; hostname`, and writes its output into a
new directory under your remote home (~/rclust-it-<random>), which is removed afterwards.
"""

import re
import shlex
import time
import uuid

import pytest

pytestmark = pytest.mark.cluster

TIMEOUT_S = 15 * 60
POLL_S = 15
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
         "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def _state(ssh, job_id):
    """(state, finished): squeue while the job is queued or running, then sacct."""
    code, out, _ = ssh.execute_command(f"squeue -h -j {job_id} -o %T")
    if code == 0 and out.strip():
        return out.split()[0], False
    code, out, _ = ssh.execute_command(f"sacct -n -X -P -j {job_id} -o State", use_login_shell=True)
    state = out.split()[0] if code == 0 and out.strip() else "UNKNOWN"  # sacct may lag a little
    return state, state in FINAL


def test_tiny_job_completes(cluster, scheduler, tmp_path, capsys):
    name = cluster.name
    with capsys.disabled():
        print(f"\n[rclust] submitting ONE tiny real job to {name} (1 CPU, 1 minute, 256M); "
              "this spends a little of your allocation.")
    tag = f"rclust-it-{uuid.uuid4().hex[:12]}"
    assert re.fullmatch(r"rclust-it-[0-9a-f]{12}", tag)  # the only thing cleanup may delete
    script = tmp_path / "rclust_it.sh"
    script.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={tag}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --time=0:01:00\n"
        "#SBATCH --mem=256M\n"
        f"#SBATCH --output={tag}/%j.out\n"  # relative to the submit directory, the remote home
        "sleep 20\n"
        "hostname\n")
    spec = scheduler.analyze(str(script))
    assert (spec.cpus, spec.time, spec.memory, spec.gpus) == (1, "0:01:00", "256M", None)

    ssh = cluster.ssh
    job, state, finished = None, "UNSUBMITTED", False
    try:
        job = scheduler.submit(str(script), cluster_name=name, remote_dir=f"~/{tag}")
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            state, finished = _state(ssh, job.job_id)
            if finished:
                break
            time.sleep(POLL_S)
        assert finished, f"job {job.job_id} on {name} did not finish within {TIMEOUT_S // 60} min ({state})"
        assert state == "COMPLETED", f"job {job.job_id} on {name} ended {state}"
        code, out, err = ssh.execute_command(f"cat -- {tag}/{job.job_id}.out")
        assert code == 0 and out.strip(), f"no output from job {job.job_id}: {err}"
    finally:
        if job is not None and not finished:
            ssh.execute_command(f"scancel {shlex.quote(job.job_id)}")
        code, _, err = ssh.execute_command(f'rm -rf -- "$HOME"/{tag}')
        with capsys.disabled():
            print(f"[rclust] {'removed' if code == 0 else 'could NOT remove'} ~/{tag} on {name}"
                  + (f": {err}" if code else ""))
