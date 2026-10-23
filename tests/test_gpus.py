"""GPU memory and MIG from a GPU's name and the config."""

import pytest

from rclust.config import ClusterConfig, ConfigError
from rclust.gpus import gpu_memory_gb, is_mig


@pytest.mark.parametrize("name, memory", [
    ("nvidia_h100_80gb_hbm3", 80), ("h100", 80), ("H100", 80), ("h200", 141),
    ("h100_1g.10gb", 10), ("nvidia_h100_80gb_hbm3_3g.40gb", 40),  # MIG: the slice's memory
    ("a100_80gb", 80), ("nvidia_a100-sxm4-40gb", 40),
    ("l40s", 48), ("l40", 48), ("a10", 24), ("t4", 16), ("rtx_a6000", 48),
    ("mi300a", 128), ("a100", None), ("v100", None), ("generic", None), ("", None),  # several sizes: the config must say
])
def test_memory_from_the_name(name, memory):
    assert gpu_memory_gb(name) == memory


def test_mig_slices():
    assert is_mig("h100_3g.40gb") and is_mig("nvidia_h100_80gb_hbm3_1g.10gb")
    assert not is_mig("nvidia_h100_80gb_hbm3") and not is_mig("a100")


def test_definitions_merge_config_and_discovered_names():
    cluster = ClusterConfig.from_dict("c", {
        "gpu_types": [
            {"type": "a100", "memory": 40},                              # memory given
            {"type": "h100_1g", "slurm_spec": "h100_1g.10gb"},           # read from the spec
            {"type": "old", "slurm_spec": "x", "memory": 20, "is_full_gpu": False},  # slurm-llm's spelling
        ],
        "resources": {"gpus": ["a100", "h100_1g.10gb", "nvidia_l40s", "generic"]},
    })
    defs = {d["type"]: d for d in cluster.gpu_definitions()}
    assert defs["a100"] == {"type": "a100", "slurm_spec": "a100", "memory": 40, "mig": False}
    assert defs["h100_1g"] == {"type": "h100_1g", "slurm_spec": "h100_1g.10gb", "memory": 10, "mig": True}
    assert defs["old"]["mig"] is True
    assert defs["nvidia_l40s"]["memory"] == 48  # discovered, not in gpu_types
    assert "generic" not in defs and "h100_1g.10gb" not in defs  # already covered by h100_1g


@pytest.mark.parametrize("bad", [{"memory": "80GB"}, {"memory": 0}, {"mig": "yes"}, {"memory": True}])
def test_invalid_gpu_details(bad):
    with pytest.raises(ConfigError, match="GPU type 'h100'"):
        ClusterConfig.from_dict("c", {"gpu_types": [{"type": "h100", **bad}]})
