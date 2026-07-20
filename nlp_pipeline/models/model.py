"""
PubMedBERT-based Hierarchical Multi-Label Clinical Text Classifier.

**Kept** from Phase 2 notebook:
  - ``[CLS]`` token contextual pooling
  - Two-layer classification head (Linear → GELU → Dropout → Linear)
  - ``LabelAttention`` mechanism (from ``AdvancedBertClassifier``)

**Modified**:
  - Default backbone switched from ``bert-base-uncased`` to PubMedBERT
  - Model returns **logits only** in ``forward()``; loss is computed
    externally by the training engine (clean separation per the plan).
  - ``LabelAttention`` is toggled via ``use_label_attention`` flag
  - Class renamed to ``ClinicalHMLTCModel``
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


# ════════════════════════════════════════════════════════════════════
# Label-Aware Attention  (kept from notebook 2 Section 5.2)
# ════════════════════════════════════════════════════════════════════

class LabelAttentionHead(nn.Module):
    """
    Label-aware attention mechanism.

    Each label has a learnable query vector that attends to the BERT
    sequence output, producing a label-specific representation.

    Kept from notebook 2 ``LabelAttention`` class.
    """

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size))
        self.attention_scale = hidden_size ** 0.5
        nn.init.xavier_uniform_(self.label_queries)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)
        attention_mask : (batch, seq_len)  — 1 for real tokens, 0 for padding

        Returns
        -------
        label_representations : (batch, num_labels, hidden_size)
        attn_weights          : (batch, num_labels, seq_len)
        """
        # Attention scores: (batch, num_labels, seq_len)
        scores = torch.matmul(
            self.label_queries.unsqueeze(0),      # (1, num_labels, hidden)
            hidden_states.transpose(1, 2),         # (batch, hidden, seq_len)
        ) / self.attention_scale

        # Mask padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        label_representations = torch.matmul(attn_weights, hidden_states)

        return label_representations, attn_weights


# ════════════════════════════════════════════════════════════════════
# Main model
# ════════════════════════════════════════════════════════════════════

class ClinicalHMLTCModel(nn.Module):
    """
    PubMedBERT + multi-label classification head.

    Architecture (default — ``use_label_attention=False``)::

        Text → PubMedBERT → [CLS] → Dropout → FC → GELU → Dropout → FC → Logits

    Architecture (advanced — ``use_label_attention=True``)::

        Text → PubMedBERT → ┬─ [CLS] ──────────────────┐
                             └─ LabelAttention ──────────┤
                                                         ↓
                                                  Concat → FC → Logits

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default is PubMedBERT.
    num_labels : int
        Number of output labels (parent + child, or flat).
    dropout_rate : float
    use_label_attention : bool
        If *True*, add the ``LabelAttentionHead`` from notebook V3 and
        fuse its output with the [CLS] representation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        num_labels: int = 50,
        dropout_rate: float = 0.3,
        use_label_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.use_label_attention = use_label_attention

        # ── Encoder ─────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size

        if use_label_attention:
            # ── V3 path: Label Attention + CLS fusion ───────────────
            self.label_attention = LabelAttentionHead(hidden_size, num_labels)
            self.dropout = nn.Dropout(dropout_rate)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, 1),
            )
        else:
            # ── V2 path: [CLS] + two-layer head ────────────────────
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 2, num_labels),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        logits : torch.Tensor of shape ``(batch, num_labels)``
            Raw logits (no sigmoid).  Loss is computed externally.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state   # (B, L, H)
        cls_output = sequence_output[:, 0, :]          # (B, H)

        if self.use_label_attention:
            label_repr, _ = self.label_attention(sequence_output, attention_mask)
            # label_repr: (B, num_labels, H)
            cls_expanded = cls_output.unsqueeze(1).expand(-1, self.num_labels, -1)
            combined = torch.cat([cls_expanded, label_repr], dim=-1)
            combined = self.dropout(combined)
            logits = self.fusion(combined).squeeze(-1)   # (B, num_labels)
        else:
            logits = self.classifier(cls_output)          # (B, num_labels)

        return logits

    # ── Convenience ─────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"ClinicalHMLTCModel("
            f"num_labels={self.num_labels}, "
            f"label_attention={self.use_label_attention}, "
            f"params={params['trainable']:,} trainable)"
        )
