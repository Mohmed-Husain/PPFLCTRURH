"""
Training engine — ``train_one_epoch``, ``validate``, ``fit``.

**Kept** from Phase 2 notebook:
  - Training loop structure (optimizer, scheduler, gradient accumulation,
    gradient clipping, best-model tracking via ``deepcopy``)
  - ``evaluate_transformer`` → renamed ``validate``

**Modified**:
  - Extracted into clean, stateless functions (no ``Config`` dependency —
    all parameters are passed as arguments).
  - ``train_one_epoch`` is the **key FL handoff function**: it takes model,
    dataloader, optimizer, loss_fn, device and returns ``(avg_loss, metrics)``.
  - ``fit`` composes ``train_one_epoch`` + ``validate`` for centralized
    baseline experiments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════
# Single-epoch training  (FL handoff function)
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    *,
    scheduler: Optional[Any] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    epoch_idx: int = 0,
    total_epochs: int = 1,
    silent: bool = False,
) -> Dict[str, float]:
    """
    Run **one** local training epoch.

    This is the function that the FL team calls per federated round.

    Parameters
    ----------
    model : nn.Module
        The ``ClinicalHMLTCModel`` (must already be on *device*).
    dataloader : DataLoader
        Training data for this epoch.
    optimizer : Optimizer
    loss_fn : nn.Module
        Any loss from ``nlp_pipeline.models.loss``.
    device : str or torch.device
    scheduler : optional LR scheduler (stepped per optimizer step)
    grad_accum_steps : int
    max_grad_norm : float
    epoch_idx, total_epochs : ints  (for progress-bar display)
    silent : bool  — suppress progress bar

    Returns
    -------
    dict with keys ``{"train_loss": float, "num_samples": int}``
    """
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    optimizer.zero_grad()

    iterator = dataloader
    if not silent:
        iterator = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{total_epochs} [Train]")

    for step, batch in enumerate(iterator):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels) / grad_accum_steps
        loss.backward()

        batch_loss = loss.item() * grad_accum_steps
        num_samples += input_ids.size(0)

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        epoch_loss += batch_loss

        if not silent:
            iterator.set_postfix({"loss": f"{batch_loss:.4f}"})

    # Handle remaining gradients if steps not divisible by accum
    if (step + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / len(dataloader)
    return {"train_loss": avg_loss, "num_samples": num_samples}


# ════════════════════════════════════════════════════════════════════
# Validation / evaluation pass
# ════════════════════════════════════════════════════════════════════

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    *,
    threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a DataLoader **without** gradient computation.

    Kept from notebook 2 ``evaluate_transformer()``.

    Returns
    -------
    avg_loss : float
    predictions : np.ndarray  (N, C)  binary
    true_labels : np.ndarray  (N, C)  binary
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, np.vstack(all_preds), np.vstack(all_labels)


# ════════════════════════════════════════════════════════════════════
# Full training loop (centralized baseline)
# ════════════════════════════════════════════════════════════════════

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    *,
    num_epochs: int = 5,
    scheduler: Optional[Any] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Full centralized training loop with validation monitoring.

    Composes ``train_one_epoch`` + ``validate`` and tracks the best
    model by validation F1 micro.

    Returns
    -------
    history : dict with keys
        ``train_loss``, ``val_loss``, ``val_f1_micro``, ``val_f1_macro``
    """
    model.to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
    }
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # ── Train ───────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scheduler=scheduler,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            epoch_idx=epoch,
            total_epochs=num_epochs,
        )
        history["train_loss"].append(train_metrics["train_loss"])

        # ── Validate ────────────────────────────────────────────
        val_loss, val_preds, val_labels = validate(
            model, val_loader, loss_fn, device, threshold=threshold,
        )
        history["val_loss"].append(val_loss)

        val_f1_micro = float(f1_score(val_labels, val_preds, average="micro", zero_division=0))
        val_f1_macro = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        history["val_f1_micro"].append(val_f1_micro)
        history["val_f1_macro"].append(val_f1_macro)

        print(
            f"  Epoch {epoch+1}: train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1_micro={val_f1_micro:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f}"
        )

        # ── Best model tracking ─────────────────────────────────
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            best_model_state = deepcopy(model.state_dict())
            print(f"    -> New best model (F1 micro: {best_val_f1:.4f})")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val F1 micro: {best_val_f1:.4f})")

    return history
