"""Read common resource directives from the Slurm script header."""

import shlex
from pathlib import Path
from .cluster import JobSpec


class JobParser:
    @staticmethod
    def parse_file(path: str) -> JobSpec:
        """
        Parses a job script file for #SBATCH directives.
        """
        content = Path(path).read_text()
        return JobParser.parse_content(content)

    @staticmethod
    def parse_content(content: str) -> JobSpec:
        spec = JobSpec()
        aliases = {"-c": "--cpus-per-task", "-G": "--gpus", "-t": "--time",
                   "-N": "--nodes", "-p": "--partition", "-n": "--ntasks"}
        fields = {"--cpus-per-task": ("cpus", int), "--mem": ("memory", str),
                  "--mem-per-cpu": ("memory_per_cpu", str), "--time": ("time", str),
                  "--nodes": ("nodes", int), "--partition": ("partition", str),
                  "--ntasks": ("tasks", int), "--dependency": ("dependency", str)}
        for line in content.splitlines():
            line = line.lstrip()
            if not line or line.startswith("#!"):
                continue
            if not line.startswith("#"):
                break  # Slurm ignores directives after the first executable line.
            if not line.startswith("#SBATCH"):
                continue
            tokens = shlex.split(line[len("#SBATCH"):], comments=True)
            i = 0
            while i < len(tokens):
                token = tokens[i]
                key, separator, value = token.partition("=")
                if not separator:
                    if len(token) > 2 and token[:2] in aliases:
                        key, value = token[:2], token[2:]
                    elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                        i += 1
                        value = tokens[i]
                key = aliases.get(key, key)
                if key in fields and value:
                    field, convert = fields[key]
                    setattr(spec, field, convert(value))
                elif key in ("--gpus", "--gpus-per-node", "--gres") and value:
                    gpu = value
                    if key == "--gres":
                        spec.gres = value
                        gpu_resources = [v[4:] for v in value.split(",") if v.startswith("gpu:")]
                        if len(gpu_resources) > 1:
                            spec.gpus = sum(int(v.rsplit(":", 1)[-1]) for v in gpu_resources)
                            spec.gpu_type = None
                            spec.gpus_per_node = True
                            i += 1
                            continue
                        gpu = gpu_resources[0] if gpu_resources else ""
                    if gpu:
                        gpu_type, _, count = gpu.rpartition(":")
                        spec.gpus = int(count or gpu)
                        spec.gpu_type = gpu_type or None
                        spec.gpus_per_node = key in ("--gpus-per-node", "--gres")
                i += 1
        return spec
