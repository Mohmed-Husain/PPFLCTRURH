"""
Configurable loss functions for hierarchical multi-label classification.

Extracted and operationalised from the scaffolded ``HierarchicalLoss``
in Phase 2 notebook Section 5.1.  Now split into composable, standalone
modules that can be used individually or combined:

  - ``FocalLoss``                — class-imbalance-aware BCE
  - ``HierarchicalConsistencyLoss`` — parent-child violation penalty
  - ``CombinedHMLTCLoss``       — composes any combination of the above

Usage
-----
Select via ``PipelineConfig.loss_type``:

    ============  ======================================
    loss_type     Result
    ============  ======================================
    ``"bce"``     Standard ``BCEWithLogitsLoss``
    ``"focal"``   ``FocalLoss``
    ``"hierarchical"``  ``BCEWithLogitsLoss`` + ``HierarchicalConsistencyLoss``
    ``"combined"``      ``FocalLoss`` + ``HierarchicalConsistencyLoss``
    ============  ======================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
# Focal Loss
# ════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification.

    Addresses class imbalance by down-weighting easy (well-classified)
    examples and focusing training on hard examples.

    Math (per element)::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Kept from notebook 2 ``HierarchicalLoss.forward`` — the focal BCE
    computation — but extracted into a standalone module.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  ``gamma=0`` reduces to standard BCE.
    alpha : float or None
        Balancing factor.  ``None`` = no alpha weighting.
    label_smoothing : float
        Smooth targets: ``y' = y*(1-eps) + eps/2``.
    reduction : str
        ``"mean"`` or ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  raw logits
        targets : (B, C)  multi-hot targets  (0 or 1, float)
        """
        # Optional label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        # Element-wise BCE (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ════════════════════════════════════════════════════════════════════
# Hierarchical Consistency Loss
# ════════════════════════════════════════════════════════════════════

class HierarchicalConsistencyLoss(nn.Module):
    """
    Penalises predictions where a child label is active but its parent
    is not.

    Kept from notebook 2 ``HierarchicalLoss`` — the hierarchy-penalty
    branch — extracted into a standalone module.

    Parameters
    ----------
    child_parent_pairs : list of (child_idx, parent_idx)
        Index pairs derived from the hierarchy.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        child_parent_pairs: List[Tuple[int, int]],
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.child_parent_pairs = child_parent_pairs
        self.reduction = reduction

    @classmethod
    def from_hierarchy(
        cls,
        child_to_parent: Dict[str, str],
        label_to_idx: Dict[str, int],
        reduction: str = "mean",
    ) -> "HierarchicalConsistencyLoss":
        """
        Build from a ``child_to_parent`` mapping and ``label_to_idx`` dict.

        This is the recommended constructor when working with
        ``CodingSystemHierarchy`` and ``HierarchicalLabelEncoder``.
        """
        pairs: List[Tuple[int, int]] = []
        for child_label, parent_label in child_to_parent.items():
            if child_label in label_to_idx and parent_label in label_to_idx:
                pairs.append((label_to_idx[child_label], label_to_idx[parent_label]))
        return cls(pairs, reduction=reduction)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C)  raw logits

        Returns
        -------
        penalty : scalar tensor
        """
        if not self.child_parent_pairs:
            return torch.tensor(0.0, device=logits.device)

        probs = torch.sigmoid(logits)
        violations = []
        for child_idx, parent_idx in self.child_parent_pairs:
            # Penalise when child_prob > parent_prob
            violation = torch.relu(probs[:, child_idx] - probs[:, parent_idx])
            violations.append(violation)

        # Stack and reduce
        violations_tensor = torch.stack(violations, dim=1)  # (B, num_pairs)
        if self.reduction == "mean":
            return violations_tensor.mean()
        return violations_tensor.sum()


# ════════════════════════════════════════════════════════════════════
# Combined Loss (configurable composition)
# ════════════════════════════════════════════════════════════════════

class CombinedHMLTCLoss(nn.Module):
    """
    Compose primary classification loss + optional hierarchical penalty.

    Parameters
    ----------
    primary_loss : nn.Module
        The main classification loss (``FocalLoss`` or ``BCEWithLogitsLoss``).
    hierarchy_loss : HierarchicalConsistencyLoss or None
        Optional hierarchy-violation penalty.
    hierarchy_weight : float
        Weight for the hierarchy penalty term.
    """

    def __init__(
        self,
        primary_loss: nn.Module,
        hierarchy_loss: Optional[HierarchicalConsistencyLoss] = None,
        hierarchy_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.primary_loss = primary_loss
        self.hierarchy_loss = hierarchy_loss
        self.hierarchy_weight = hierarchy_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)
        targets : (B, C)
        """
        loss = self.primary_loss(logits, targets)

        if self.hierarchy_loss is not None and self.hierarchy_weight > 0:
            h_penalty = self.hierarchy_loss(logits)
            loss = loss + self.hierarchy_weight * h_penalty

        return loss


# ════════════════════════════════════════════════════════════════════
# Factory — build loss from config
# ════════════════════════════════════════════════════════════════════

def build_loss(
    loss_type: str,
    *,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    label_smoothing: float = 0.0,
    hierarchy_penalty_weight: float = 0.5,
    child_to_parent: Optional[Dict[str, str]] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> nn.Module:
    """
    Factory function: create the appropriate loss module from a string key.

    Parameters
    ----------
    loss_type : str
        One of ``"bce"``, ``"focal"``, ``"hierarchical"``, ``"combined"``.
    child_to_parent, label_to_idx
        Required when *loss_type* includes a hierarchy component.

    Returns
    -------
    nn.Module
    """
    # Primary loss
    if loss_type in ("bce", "hierarchical"):
        primary: nn.Module = nn.BCEWithLogitsLoss()
    elif loss_type in ("focal", "combined"):
        primary = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}. "
            f"Choose from: 'bce', 'focal', 'hierarchical', 'combined'."
        )

    # Hierarchy penalty
    h_loss: Optional[HierarchicalConsistencyLoss] = None
    if loss_type in ("hierarchical", "combined"):
        if child_to_parent is None or label_to_idx is None:
            raise ValueError(
                f"loss_type={loss_type!r} requires 'child_to_parent' and "
                f"'label_to_idx' arguments to build the hierarchy penalty."
            )
        h_loss = HierarchicalConsistencyLoss.from_hierarchy(
            child_to_parent, label_to_idx
        )

    if h_loss is not None:
        return CombinedHMLTCLoss(primary, h_loss, hierarchy_penalty_weight)

    return primary
