"""What a GPU's name says about it: its memory, and whether it is a MIG slice.

Slurm names GPUs by site convention ("h100", "nvidia_h100_80gb_hbm3", "h100_3g.40gb"), so the
memory is read from the name where it is spelled out (MIG slices always are), or else from the
model when every variant of it has the same memory. Models sold with several memory sizes
(A100: 40 or 80 GB, V100: 16 or 32 GB) give None: only the site's config can say which it has.
"""

import re
from typing import Optional

MIG = re.compile(r"(\d+)g\.(\d+)gb", re.IGNORECASE)  # e.g. 3g.40gb: 3 compute slices, 40 GB
_SPELLED_OUT = re.compile(r"(?:^|[^0-9])(\d{2,3})gb(?:[^a-z]|$)", re.IGNORECASE)

# Models whose every variant has the same memory (GB). Checked in order, as whole words of the
# name, so "l40s" is not read as "l40" and "a10" is not found in "a100".
_MODELS = [
    ("gh200", 96), ("h200", 141), ("h100", 80),
    ("l40s", 48), ("l40", 48), ("a40", 48), ("l4", 24),
    ("rtx_a6000", 48), ("a6000", 48), ("rtx_a5000", 24), ("a5000", 24), ("a4500", 20), ("a4000", 16),
    ("rtx8000", 48), ("rtx_8000", 48), ("a10", 24), ("t4", 16),
    ("mi300x", 192), ("mi300a", 128), ("mi250x", 128), ("mi250", 128), ("mi210", 64),
]


def is_mig(name: str) -> bool:
    """Whether the name is that of a MIG slice (which cannot be combined for multi-GPU work)."""
    return bool(MIG.search(name or ""))


def gpu_memory_gb(name: str) -> Optional[int]:
    """Memory (GB) of one GPU of this name, or None when the name does not say."""
    name = (name or "").lower()
    mig = MIG.search(name)
    if mig:
        return int(mig.group(2))
    spelled = _SPELLED_OUT.search(name)
    if spelled:
        return int(spelled.group(1))
    for model, memory in _MODELS:
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(model)}(?:$|[^a-z0-9])", name):
            return memory
    return None
