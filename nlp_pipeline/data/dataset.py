"""
PyTorch Dataset for hierarchical multi-label clinical text classification.

**Kept** from Phase 2 notebook ``MultiLabelTextDataset``:
  - ``__getitem__`` returns ``{input_ids, attention_mask, labels}``
  - Tokenisation with HuggingFace tokenizer
  - Label tensor conversion

**Modified**:
  - Renamed to ``ClinicalTextDataset``
  - Added ``from_records`` classmethod for standardised record ingestion
  - Accepts optional pre-computed label matrices
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nlp_pipeline.data.label_encoder import HierarchicalLabelEncoder


class ClinicalTextDataset(Dataset):
    """
    PyTorch Dataset for multi-label text classification.

    Each sample returns::

        {
            "input_ids":      LongTensor  (max_length,),
            "attention_mask": LongTensor  (max_length,),
            "labels":         FloatTensor (num_labels,),
        }

    Parameters
    ----------
    texts : list[str]
        Raw text strings.
    label_matrix : np.ndarray
        Multi-hot label matrix of shape ``(N, num_labels)``.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer (e.g. PubMedBERT).
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        texts: List[str],
        label_matrix: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        assert len(texts) == label_matrix.shape[0], (
            f"texts ({len(texts)}) and labels ({label_matrix.shape[0]}) "
            f"must have the same number of samples."
        )
        self.texts = texts
        self.labels = label_matrix.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.FloatTensor(label),
        }

    # ── Convenience constructors ────────────────────────────────────

    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        label_encoder: HierarchicalLabelEncoder,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        text_key: str = "text",
        label_key: str = "labels",
    ) -> "ClinicalTextDataset":
        """
        Build a dataset from the standardised record format::

            {"text": "...", "labels": ["LabelA", "LabelB"]}

        This is the format produced by every ``BaseDataPlugin``.
        """
        texts = [r[text_key] for r in records]
        raw_labels = [r[label_key] for r in records]
        label_matrix = label_encoder.encode(raw_labels)
        return cls(texts, label_matrix, tokenizer, max_length)

    def __repr__(self) -> str:
        return (
            f"ClinicalTextDataset(samples={len(self)}, "
            f"num_labels={self.labels.shape[1]}, "
            f"max_length={self.max_length})"
        )
