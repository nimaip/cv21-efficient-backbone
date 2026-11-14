from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def git_commit(root: Path | None = None) -> str:
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit.strip()
    root = root or PROJECT_ROOT
    stamp = root / ".git_commit"
    if stamp.is_file():
        text = stamp.read_text().strip()
        if text:
            return text
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def param_groups(module: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrix weights. Leave biases, norms, and position tokens undecayed."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        skip_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        )
        if skip_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
