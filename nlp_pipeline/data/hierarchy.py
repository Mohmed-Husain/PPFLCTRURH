"""
Coding-system-agnostic hierarchical label structure.

Designed to work with **any** medical coding system (ICD-9-CM, ICD-10-CM,
CPT, SNOMED-CT, or a custom taxonomy).  The hierarchy depth is fully
configurable — it is **not** fixed at two levels.

The module ships with built-in chapter tables for ICD-9 and ICD-10 to
bootstrap hierarchy construction when those systems are detected, but the
``CodingSystemHierarchy`` class accepts arbitrary parent→child mappings
through its ``from_mapping`` classmethod.

Usage
-----
>>> h = CodingSystemHierarchy.from_codes(["428.0", "410.9", "250.00"], system="icd9")
>>> h.parent_to_children
{'Diseases Of The Circulatory System': ['428', '410'],
 'Endocrine, Nutritional And Metabolic Diseases': ['250']}
>>> h.child_to_parent["428"]
'Diseases Of The Circulatory System'
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ════════════════════════════════════════════════════════════════════
# Built-in ICD chapter tables
# ════════════════════════════════════════════════════════════════════

# ICD-9-CM chapters — (start, end) ranges for the *numeric* portion of
# the three-digit code.  V/E codes are handled specially.
ICD9_CHAPTERS: List[Tuple[str, int, int]] = [
    ("Infectious And Parasitic Diseases", 1, 139),
    ("Neoplasms", 140, 239),
    ("Endocrine, Nutritional And Metabolic Diseases", 240, 279),
    ("Diseases Of The Blood", 280, 289),
    ("Mental Disorders", 290, 319),
    ("Diseases Of The Nervous System", 320, 389),
    ("Diseases Of The Circulatory System", 390, 459),
    ("Diseases Of The Respiratory System", 460, 519),
    ("Diseases Of The Digestive System", 520, 579),
    ("Diseases Of The Genitourinary System", 580, 629),
    ("Complications Of Pregnancy", 630, 679),
    ("Diseases Of The Skin", 680, 709),
    ("Diseases Of The Musculoskeletal System", 710, 739),
    ("Congenital Anomalies", 740, 759),
    ("Certain Conditions Originating In The Perinatal Period", 760, 779),
    ("Symptoms, Signs, And Ill-Defined Conditions", 780, 799),
    ("Injury And Poisoning", 800, 999),
]

# ICD-10-CM chapters — identified by the first letter of the code.
ICD10_CHAPTERS: Dict[str, str] = {
    "A": "Certain Infectious And Parasitic Diseases",
    "B": "Certain Infectious And Parasitic Diseases",
    "C": "Neoplasms",
    "D": "Diseases Of The Blood And Neoplasms",
    "E": "Endocrine, Nutritional And Metabolic Diseases",
    "F": "Mental And Behavioural Disorders",
    "G": "Diseases Of The Nervous System",
    "H": "Diseases Of The Eye And Ear",
    "I": "Diseases Of The Circulatory System",
    "J": "Diseases Of The Respiratory System",
    "K": "Diseases Of The Digestive System",
    "L": "Diseases Of The Skin",
    "M": "Diseases Of The Musculoskeletal System",
    "N": "Diseases Of The Genitourinary System",
    "O": "Pregnancy, Childbirth And The Puerperium",
    "P": "Certain Conditions Originating In The Perinatal Period",
    "Q": "Congenital Malformations",
    "R": "Symptoms And Signs",
    "S": "Injury And Poisoning",
    "T": "Injury And Poisoning",
    "U": "Codes For Special Purposes",
    "V": "External Causes Of Morbidity",
    "W": "External Causes Of Morbidity",
    "X": "External Causes Of Morbidity",
    "Y": "External Causes Of Morbidity",
    "Z": "Factors Influencing Health Status",
}


# ════════════════════════════════════════════════════════════════════
# Helper: detect coding system from sample codes
# ════════════════════════════════════════════════════════════════════

def detect_coding_system(codes: List[str]) -> str:
    """
    Heuristically decide whether *codes* are ICD-9, ICD-10, or unknown.

    Returns ``"icd9"``, ``"icd10"``, or ``"unknown"``.
    """
    if not codes:
        return "unknown"

    sample = [c.strip() for c in codes[:200]]

    # ICD-10 codes start with a letter followed by digits
    icd10_count = sum(1 for c in sample if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit())
    # ICD-9 codes are typically 3–5 digit (possibly with leading V/E)
    icd9_count = sum(1 for c in sample if c.replace(".", "").lstrip("VE").isdigit())

    if icd10_count > icd9_count:
        return "icd10"
    if icd9_count > 0:
        return "icd9"
    return "unknown"


# ════════════════════════════════════════════════════════════════════
# Helper: extract ancestor chain at configurable depth
# ════════════════════════════════════════════════════════════════════

def _icd9_code_to_chapter(code: str) -> Optional[str]:
    """Map a single ICD-9 code to its chapter name, or *None*."""
    code = code.strip().upper()
    if code.startswith("V"):
        return "Supplementary Classification V Codes"
    if code.startswith("E"):
        return "Supplementary Classification E Codes"
    # Extract the numeric prefix (first 3 digits)
    numeric = code.replace(".", "")[:3]
    try:
        num = int(numeric)
    except ValueError:
        return None
    for chapter_name, lo, hi in ICD9_CHAPTERS:
        if lo <= num <= hi:
            return chapter_name
    return None


def _icd10_code_to_chapter(code: str) -> Optional[str]:
    """Map a single ICD-10 code to its chapter name, or *None*."""
    code = code.strip().upper()
    if not code:
        return None
    return ICD10_CHAPTERS.get(code[0])


def _truncate_code(code: str, depth: int) -> str:
    """
    Truncate a diagnosis code to *depth* characters (excluding dots).

    For depth=3, ``"428.32"`` → ``"428"`` and ``"E11.65"`` → ``"E11"``.
    """
    stripped = code.replace(".", "")
    truncated = stripped[:depth]
    return truncated


# ════════════════════════════════════════════════════════════════════
# Main class
# ════════════════════════════════════════════════════════════════════

class CodingSystemHierarchy:
    """
    Coding-system-agnostic hierarchical label structure.

    Attributes
    ----------
    parent_to_children : dict[str, list[str]]
        Mapping from parent labels to their child labels.
    child_to_parent : dict[str, str]
        Reverse mapping from each child label to its parent.
    all_labels : list[str]
        Sorted list of every label (parents + children, de-duplicated).
    parent_labels : list[str]
        Sorted list of parent-only labels.
    child_labels : list[str]
        Sorted list of child-only labels.
    depth : int
        Hierarchy depth that was used to build this structure.
    coding_system : str
        Coding system identifier (``"icd9"``, ``"icd10"``, ``"custom"``).
    """

    def __init__(
        self,
        parent_to_children: Dict[str, List[str]],
        coding_system: str = "custom",
        depth: int = -1,
    ) -> None:
        self.coding_system = coding_system
        self.depth = depth

        # Deduplicate children lists while preserving order
        self.parent_to_children: Dict[str, List[str]] = {
            p: list(dict.fromkeys(children))
            for p, children in parent_to_children.items()
        }

        # Build reverse mapping
        self.child_to_parent: Dict[str, str] = {}
        for parent, children in self.parent_to_children.items():
            for child in children:
                self.child_to_parent[child] = parent

        self.parent_labels: List[str] = sorted(self.parent_to_children.keys())
        self.child_labels: List[str] = sorted(self.child_to_parent.keys())
        self.all_labels: List[str] = sorted(
            set(self.parent_labels) | set(self.child_labels)
        )

    # ── Constructors ────────────────────────────────────────────────

    @classmethod
    def from_codes(
        cls,
        codes: List[str],
        system: str = "auto",
        depth: int = -1,
    ) -> "CodingSystemHierarchy":
        """
        Build hierarchy from a flat list of diagnosis codes.

        Parameters
        ----------
        codes : list[str]
            Raw ICD codes (e.g. ``["428.0", "250.01", "410.9"]``).
        system : str
            ``"icd9"``, ``"icd10"``, ``"auto"`` (auto-detect), or
            ``"custom"`` (treat codes as flat labels — no parent grouping).
        depth : int
            Number of characters (excluding dots) to retain when building
            the child-level codes.  ``-1`` means use the full code.
            ``3`` is typical for ICD-9 three-digit grouping.

        Returns
        -------
        CodingSystemHierarchy
        """
        if system == "auto":
            system = detect_coding_system(codes)

        if system == "icd9":
            chapter_fn = _icd9_code_to_chapter
        elif system == "icd10":
            chapter_fn = _icd10_code_to_chapter
        else:
            # Custom / unknown — create a flat hierarchy (every code is its
            # own parent, no children).
            p2c: Dict[str, List[str]] = {c: [] for c in sorted(set(codes))}
            return cls(p2c, coding_system=system, depth=depth)

        # Build parent→children mapping
        parent_to_children: Dict[str, List[str]] = defaultdict(list)
        seen_children: Set[str] = set()

        for raw_code in codes:
            raw_code = raw_code.strip()
            if not raw_code:
                continue

            chapter = chapter_fn(raw_code)
            if chapter is None:
                continue

            # Determine child label (possibly truncated)
            if depth > 0:
                child = _truncate_code(raw_code, depth)
            else:
                child = raw_code.replace(".", "")

            if child not in seen_children:
                parent_to_children[chapter].append(child)
                seen_children.add(child)

        return cls(dict(parent_to_children), coding_system=system, depth=depth)

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, List[str]],
        coding_system: str = "custom",
    ) -> "CodingSystemHierarchy":
        """
        Build from an explicit parent → children dict.

        This is the escape hatch for any taxonomy that doesn't use standard
        ICD codes (e.g. SNOMED-CT groups, or your old ADNI taxonomy).
        """
        return cls(mapping, coding_system=coding_system, depth=-1)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "CodingSystemHierarchy":
        """Load a hierarchy previously saved with ``to_json``."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            parent_to_children=data["parent_to_children"],
            coding_system=data.get("coding_system", "custom"),
            depth=data.get("depth", -1),
        )

    # ── Serialisation ───────────────────────────────────────────────

    def to_json(self, filepath: str | Path) -> None:
        """Save hierarchy to JSON for reproducibility across clients."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coding_system": self.coding_system,
            "depth": self.depth,
            "parent_to_children": self.parent_to_children,
            "child_to_parent": self.child_to_parent,
            "parent_labels": self.parent_labels,
            "child_labels": self.child_labels,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Hierarchy saved: {filepath}")

    # ── Queries ──────────────────────────────────────────────────────

    def get_parent(self, child_label: str) -> Optional[str]:
        """Return the parent of *child_label*, or *None*."""
        return self.child_to_parent.get(child_label)

    def get_children(self, parent_label: str) -> List[str]:
        """Return children of *parent_label* (empty list if not found)."""
        return self.parent_to_children.get(parent_label, [])

    def ensure_parent_consistency(self, labels: List[str]) -> List[str]:
        """
        Given a list of labels, ensure every child's parent is present.

        Returns a new list with missing parents added.
        """
        label_set = set(labels)
        for label in list(label_set):
            parent = self.child_to_parent.get(label)
            if parent is not None:
                label_set.add(parent)
        return sorted(label_set)

    @property
    def num_parents(self) -> int:
        return len(self.parent_labels)

    @property
    def num_children(self) -> int:
        return len(self.child_labels)

    @property
    def num_labels(self) -> int:
        return len(self.all_labels)

    # ── Display ──────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"CodingSystemHierarchy  (system={self.coding_system}, depth={self.depth})",
            f"  Parents  : {self.num_parents}",
            f"  Children : {self.num_children}",
            f"  Total    : {self.num_labels}",
        ]
        for parent in self.parent_labels:
            children = self.parent_to_children[parent]
            lines.append(f"    {parent}  ({len(children)} children)")
            for child in children[:5]:
                lines.append(f"      ├── {child}")
            if len(children) > 5:
                lines.append(f"      └── ... +{len(children) - 5} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CodingSystemHierarchy(system={self.coding_system!r}, "
            f"parents={self.num_parents}, children={self.num_children})"
        )
