"""
Centralized configuration for the NLP pipeline.

Merges parameters from both Phase 1 (data generation) and Phase 2 (training)
notebooks into a single dataclass-based config.  Every module receives its
parameters from this config; no hardcoded hyperparameters elsewhere.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    """
    Centralized, immutable-by-convention configuration.

    Sections
    --------
    - Data paths & ingestion
    - Text preprocessing / chunking  (kept from Phase 1)
    - Coding-system hierarchy
    - Tokenizer / model architecture
    - Loss function
    - Training hyperparameters
    - Runtime
    """

    # ── Data paths ──────────────────────────────────────────────────────
    data_dir: Path = Path("data_generation")
    output_dir: Path = Path("model_outputs")

    # ── Text preprocessing / chunking (from Phase 1 notebook) ──────────
    chunk_size: int = 512           # target words per chunk
    chunk_overlap: int = 64         # overlap in words
    min_chunk_size: int = 50        # discard chunks smaller than this

    # ── Coding-system hierarchy ─────────────────────────────────────────
    #    System-agnostic: the hierarchy module interprets these at runtime.
    coding_system: str = "auto"     # "icd9", "icd10", "custom", or "auto"
    hierarchy_depth: int = -1       # -1 = unlimited, 2 = two-level, etc.
    hierarchy_file: Optional[str] = None  # optional JSON defining custom hierarchy

    # ── Tokenizer / model ───────────────────────────────────────────────
    model_name: str = (
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    max_seq_length: int = 512
    dropout_rate: float = 0.3
    use_label_attention: bool = False  # toggle LabelAttention head (V3)

    # ── Loss function ───────────────────────────────────────────────────
    #    Choices: "bce", "focal", "hierarchical", "combined"
    loss_type: str = "combined"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None  # None = no alpha weighting
    hierarchy_penalty_weight: float = 0.5
    label_smoothing: float = 0.1

    # ── Training hyperparameters ────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    classification_threshold: float = 0.5

    # ── Dataset splits ──────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Runtime ─────────────────────────────────────────────────────────
    random_seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0
    pin_memory: bool = True
    mixed_precision: bool = False
    log_every_n_steps: int = 10

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def resolve_paths(self) -> None:
        """Ensure output directories exist."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            "PipelineConfig",
            f"  model           : {self.model_name}",
            f"  device          : {self.device}",
            f"  loss            : {self.loss_type}",
            f"  coding_system   : {self.coding_system}",
            f"  hierarchy_depth : {self.hierarchy_depth}",
            f"  batch_size      : {self.batch_size} (effective {self.effective_batch_size})",
            f"  lr              : {self.learning_rate}",
            f"  epochs          : {self.num_epochs}",
            f"  max_seq_length  : {self.max_seq_length}",
            f"  dropout         : {self.dropout_rate}",
            f"  label_attention : {self.use_label_attention}",
        ]
        return "\n".join(lines)
