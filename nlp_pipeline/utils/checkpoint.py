"""
Model checkpoint save / load utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str | Path,
    *,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer or None
    epoch : int
    metrics : dict   – validation metrics at this checkpoint
    filepath : Path
    scheduler : optional LR scheduler
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, filepath)
    print(f"  ✓ Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Restore a training checkpoint.

    Returns the ``metrics`` dict stored in the checkpoint so the caller
    can decide whether to keep this as best.
    """
    filepath = Path(filepath)
    ckpt = torch.load(filepath, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"  ✓ Checkpoint loaded: {filepath}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def find_best_checkpoint(directory: str | Path, metric_key: str = "val_f1_micro") -> Optional[Path]:
    """
    Scan *directory* for ``*.pt`` checkpoints and return the path to the
    one with the highest value of *metric_key*.
    """
    directory = Path(directory)
    best_path = None
    best_score = -1.0

    for ckpt_path in sorted(directory.glob("*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            score = ckpt.get("metrics", {}).get(metric_key, -1.0)
            if score > best_score:
                best_score = score
                best_path = ckpt_path
        except Exception:
            continue

    return best_path
