"""
Public NLP API and Federated Learning client interface.

This module is the **single integration surface** between the NLP pipeline
and the Federated Learning framework.  The FL team (Ratan & Harshit) should
only need to interact with ``FederatedNLPClient``.

Design decisions
----------------
- The client exposes **only** the APIs required for local training and
  model-weight exchange: ``get_weights``, ``set_weights``, ``local_train_epoch``,
  ``evaluate``, ``get_num_samples``.
- All FL-specific optimization (FedAvg weighting, FedProx μ, SCAFFOLD
  corrections, differential-privacy noise injection) is **not** handled
  here.  Those are the FL team's responsibility and can be applied
  externally around the ``get_weights`` / ``set_weights`` boundary.
- The client wraps ``ClinicalHMLTCModel`` + ``CombinedHMLTCLoss`` +
  ``AdamW`` optimizer internally so the FL team doesn't need to
  understand the model internals.

Standalone convenience functions (``initialize_model``, ``preprocess``,
``build_dataloaders``, etc.) are also provided for centralized experiments
and scripting.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from nlp_pipeline.configs.config import PipelineConfig
from nlp_pipeline.data.data_plugin import BaseDataPlugin
from nlp_pipeline.data.dataset import ClinicalTextDataset
from nlp_pipeline.data.hierarchy import CodingSystemHierarchy
from nlp_pipeline.data.label_encoder import HierarchicalLabelEncoder
from nlp_pipeline.data.preprocess import clean_clinical_text
from nlp_pipeline.models.loss import build_loss
from nlp_pipeline.models.model import ClinicalHMLTCModel
from nlp_pipeline.training.metrics import evaluate_multilabel
from nlp_pipeline.training.trainer import train_one_epoch, validate


# ════════════════════════════════════════════════════════════════════
# Federated NLP Client
# ════════════════════════════════════════════════════════════════════

class FederatedNLPClient:
    """
    Encapsulates the NLP model, loss, optimizer, and tokenizer for one
    federated client.  Exposes the exact handoff ports the FL team needs.

    Usage (FL side)::

        client = FederatedNLPClient(config, hierarchy, label_encoder)
        client.set_weights(global_state_dict)        # receive global model
        metrics = client.local_train_epoch(loader)    # one local epoch
        updated_weights = client.get_weights()        # send back to server
        n = client.get_num_samples()                  # for FedAvg weighting

    Parameters
    ----------
    config : PipelineConfig
    hierarchy : CodingSystemHierarchy
    label_encoder : HierarchicalLabelEncoder
    """

    def __init__(
        self,
        config: PipelineConfig,
        hierarchy: CodingSystemHierarchy,
        label_encoder: HierarchicalLabelEncoder,
    ) -> None:
        self.config = config
        self.hierarchy = hierarchy
        self.label_encoder = label_encoder
        self.device = torch.device(config.device)

        # ── Model ───────────────────────────────────────────────
        self.model = ClinicalHMLTCModel(
            model_name=config.model_name,
            num_labels=label_encoder.num_labels,
            dropout_rate=config.dropout_rate,
            use_label_attention=config.use_label_attention,
        ).to(self.device)

        # ── Loss ────────────────────────────────────────────────
        self.loss_fn: nn.Module = build_loss(
            loss_type=config.loss_type,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            label_smoothing=config.label_smoothing,
            hierarchy_penalty_weight=config.hierarchy_penalty_weight,
            child_to_parent=hierarchy.child_to_parent,
            label_to_idx=label_encoder.label_to_idx,
        )

        # ── Tokenizer ──────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # ── Optimizer (created lazily or on first train) ───────
        self._optimizer: Optional[AdamW] = None
        self._scheduler: Optional[Any] = None
        self._num_samples: int = 0

    # ═══════════════════════════════════════════════════════════
    #  FL Handoff Ports
    # ═══════════════════════════════════════════════════════════

    def get_weights(self) -> OrderedDict:
        """
        Export model ``state_dict`` for aggregation.

        The FL server calls this after ``local_train_epoch`` to retrieve
        the updated weights.
        """
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
        )

    def set_weights(self, state_dict: OrderedDict) -> None:
        """
        Load global model weights received from the FL server.

        Call this **before** ``local_train_epoch`` each round.
        """
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        # Reset optimizer so momentum buffers align with new weights
        self._optimizer = None
        self._scheduler = None

    def local_train_epoch(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run **one** local training epoch.

        This is the core function the FL framework calls per round.

        Parameters
        ----------
        dataloader : DataLoader
            Client-local training data.
        **kwargs
            Forwarded to ``train_one_epoch`` (e.g. ``silent=True``).

        Returns
        -------
        dict with ``{"train_loss": float, "num_samples": int}``
        """
        # Lazy-init optimizer
        if self._optimizer is None:
            self._optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        self._num_samples = len(dataloader.dataset)

        metrics = train_one_epoch(
            model=self.model,
            dataloader=dataloader,
            optimizer=self._optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            grad_accum_steps=self.config.grad_accum_steps,
            max_grad_norm=self.config.max_grad_norm,
            **kwargs,
        )
        return metrics

    def get_num_samples(self) -> int:
        """
        Return the number of samples in the client's training set.

        Needed by FedAvg for proportional aggregation weighting.
        """
        return self._num_samples

    # ═══════════════════════════════════════════════════════════
    #  Evaluation & Inference
    # ═══════════════════════════════════════════════════════════

    def evaluate(
        self,
        dataloader: DataLoader,
        model_name: str = "FederatedClient",
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a DataLoader and return the full metric dict.

        Parameters
        ----------
        dataloader : DataLoader
        model_name : str   for the evaluation report header

        Returns
        -------
        dict   (same structure as ``evaluate_multilabel``)
        """
        val_loss, preds, labels = validate(
            self.model, dataloader, self.loss_fn, self.device,
            threshold=self.config.classification_threshold,
        )
        metrics = evaluate_multilabel(
            labels, preds, self.label_encoder.label_names,
            model_name=model_name, print_report=False,
        )
        metrics["val_loss"] = val_loss
        return metrics

    def predict(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Run inference on raw text inputs.

        Parameters
        ----------
        texts : list[str]
        batch_size : int

        Returns
        -------
        np.ndarray  (N, num_labels)  — predicted probabilities
        """
        self.model.eval()
        all_probs: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict_labels(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[List[str]]:
        """
        Predict and decode to label strings.

        Returns
        -------
        list[list[str]]  — decoded label lists per sample
        """
        probs = self.predict(texts, batch_size)
        return self.label_encoder.decode(probs, self.config.classification_threshold)

    # ═══════════════════════════════════════════════════════════
    #  DataLoader Construction
    # ═══════════════════════════════════════════════════════════

    def build_dataloader(
        self,
        records: List[Dict[str, Any]],
        shuffle: bool = True,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Build a ``DataLoader`` from standardised records.

        Parameters
        ----------
        records : list[dict]
            Each dict must have ``"text"`` and ``"labels"`` keys.
        shuffle : bool
        batch_size : int or None  (defaults to ``config.batch_size``)
        """
        dataset = ClinicalTextDataset.from_records(
            records=records,
            label_encoder=self.label_encoder,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    # ═══════════════════════════════════════════════════════════
    #  Checkpoint helpers
    # ═══════════════════════════════════════════════════════════

    def save_model(self, filepath: str) -> None:
        """Save model state_dict to disk."""
        torch.save(self.model.state_dict(), filepath)
        print(f"  ✓ Model saved: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model state_dict from disk."""
        state = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"  ✓ Model loaded: {filepath}")

    def __repr__(self) -> str:
        return (
            f"FederatedNLPClient("
            f"model={self.config.model_name!r}, "
            f"labels={self.label_encoder.num_labels}, "
            f"device={self.device})"
        )


# ════════════════════════════════════════════════════════════════════
# Standalone convenience functions (for centralized experiments)
# ════════════════════════════════════════════════════════════════════

def initialize_model(config: PipelineConfig, num_labels: int) -> ClinicalHMLTCModel:
    """Create a ``ClinicalHMLTCModel`` from config."""
    return ClinicalHMLTCModel(
        model_name=config.model_name,
        num_labels=num_labels,
        dropout_rate=config.dropout_rate,
        use_label_attention=config.use_label_attention,
    )


def preprocess(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean text in a list of standardised records."""
    return [
        {**r, "text": clean_clinical_text(r.get("text", ""))}
        for r in records
    ]


def build_dataloaders(
    train_records: List[Dict],
    val_records: List[Dict],
    test_records: List[Dict],
    label_encoder: HierarchicalLabelEncoder,
    tokenizer: Any,
    config: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from record lists."""
    def _make(records: List[Dict], shuffle: bool) -> DataLoader:
        ds = ClinicalTextDataset.from_records(
            records, label_encoder, tokenizer, config.max_seq_length,
        )
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return _make(train_records, True), _make(val_records, False), _make(test_records, False)
