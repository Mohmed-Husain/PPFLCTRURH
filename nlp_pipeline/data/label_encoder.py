"""
Hierarchical multi-hot label encoder.

Wraps ``sklearn.preprocessing.MultiLabelBinarizer`` and adds:
  - automatic parent-consistency enforcement
  - save / load for federated reproducibility
  - decode helpers

The multi-hot encoding logic is **kept** from Phase 2 notebook
(``extract_texts_and_labels`` / ``mlb``), but elevated into a reusable class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from nlp_pipeline.data.hierarchy import CodingSystemHierarchy


class HierarchicalLabelEncoder:
    """
    Multi-hot encoder that respects a hierarchical label structure.

    Parameters
    ----------
    hierarchy : CodingSystemHierarchy
        The parent↔child label structure.
    enforce_consistency : bool
        If *True*, ``encode`` will auto-add missing parent labels when a
        child is present (same logic as Phase 1 ``validate_labels``).
    """

    def __init__(
        self,
        hierarchy: CodingSystemHierarchy,
        enforce_consistency: bool = True,
    ) -> None:
        self.hierarchy = hierarchy
        self.enforce_consistency = enforce_consistency

        # Deterministic sorted label list — must be identical across all
        # federated clients for the label vectors to align.
        self.label_names: List[str] = list(hierarchy.all_labels)
        self.num_labels: int = len(self.label_names)

        # Index mappings
        self.label_to_idx: Dict[str, int] = {
            label: idx for idx, label in enumerate(self.label_names)
        }
        self.idx_to_label: Dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }

        # Fit the sklearn binarizer once
        self._mlb = MultiLabelBinarizer(classes=self.label_names)
        self._mlb.fit([self.label_names])

    # ── Encode / decode ────────────────────────────────────────────

    def encode(self, label_lists: List[List[str]]) -> np.ndarray:
        """
        Encode a batch of label lists into a multi-hot matrix.

        Parameters
        ----------
        label_lists : list[list[str]]
            Each inner list contains label strings for one sample.

        Returns
        -------
        np.ndarray of shape ``(N, num_labels)`` with dtype ``float32``.
        """
        if self.enforce_consistency:
            label_lists = [
                self.hierarchy.ensure_parent_consistency(labels)
                for labels in label_lists
            ]

        return self._mlb.transform(label_lists).astype(np.float32)

    def encode_single(self, labels: List[str]) -> np.ndarray:
        """Encode a single sample's labels → 1-D array."""
        return self.encode([labels])[0]

    def decode(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
    ) -> List[List[str]]:
        """
        Decode a probability / binary matrix back to label lists.

        Parameters
        ----------
        predictions : np.ndarray
            Shape ``(N, num_labels)`` — probabilities or binary 0/1.
        threshold : float
            Applied when *predictions* are continuous.
        """
        binary = (predictions >= threshold).astype(int)
        return [list(row) for row in self._mlb.inverse_transform(binary)]

    def decode_single(
        self,
        prediction: np.ndarray,
        threshold: float = 0.5,
    ) -> List[str]:
        """Decode a single 1-D prediction vector."""
        return self.decode(prediction.reshape(1, -1), threshold)[0]

    # ── Serialisation ──────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """
        Save label mappings to JSON.

        Must be loaded on every federated client to ensure identical
        label ordering.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_names": self.label_names,
            "label_to_idx": self.label_to_idx,
            "num_labels": self.num_labels,
            "enforce_consistency": self.enforce_consistency,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Label encoder saved: {filepath}")

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        hierarchy: CodingSystemHierarchy,
    ) -> "HierarchicalLabelEncoder":
        """Restore from a JSON file previously created by ``save``."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        encoder = cls(
            hierarchy=hierarchy,
            enforce_consistency=data.get("enforce_consistency", True),
        )
        # Verify that label ordering is consistent
        if encoder.label_names != data["label_names"]:
            raise ValueError(
                "Label name ordering mismatch between saved encoder and "
                "current hierarchy.  This would cause mis-aligned label "
                "vectors across federated clients."
            )
        return encoder

    # ── Display ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HierarchicalLabelEncoder(num_labels={self.num_labels}, "
            f"consistency={self.enforce_consistency})"
        )
