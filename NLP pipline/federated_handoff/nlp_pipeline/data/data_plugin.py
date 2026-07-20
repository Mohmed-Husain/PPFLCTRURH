"""
Abstract data ingestion interface (plugin pattern).

Defines ``BaseDataPlugin`` — the contract that every data source must
implement — and ships two concrete plugins:

  - ``JSONDataPlugin``  – loads Phase 1 JSON outputs (backward-compatible)
  - ``MIMICDataPlugin`` – placeholder for MIMIC-IV discharge notes + ICD codes

The NLP pipeline never knows *where* the data comes from; it only sees
standardised records::

    {"patient_id": ..., "admission_id": ..., "text": "...", "labels": [...]}
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp_pipeline.data.preprocess import clean_clinical_text


# ════════════════════════════════════════════════════════════════════
# Base class
# ════════════════════════════════════════════════════════════════════

class BaseDataPlugin(ABC):
    """
    Abstract base for all data-ingestion plugins.

    Subclasses must implement ``load_records`` which returns a list of
    standardised record dicts.
    """

    @abstractmethod
    def load_records(self) -> List[Dict[str, Any]]:
        """
        Load and return standardised records.

        Each record **must** contain at least::

            {
                "text":   str,    # clinical note or text chunk
                "labels": list,   # list of label strings
            }

        Optional fields (used for provenance tracking)::

            {
                "patient_id":   str | int,
                "admission_id": str | int,
            }
        """
        ...


# ════════════════════════════════════════════════════════════════════
# Concrete: JSON files from Phase 1
# ════════════════════════════════════════════════════════════════════

class JSONDataPlugin(BaseDataPlugin):
    """
    Load records from Phase 1 JSON outputs (``final_dataset.json``,
    ``train.json``, ``val.json``, ``test.json``).

    This provides **backward compatibility** with the existing notebooks.

    Parameters
    ----------
    filepath : str | Path
        Path to a JSON file containing a list of records.
    text_key : str
        Key for the text field in each record.
    label_key : str
        Key for the labels field in each record.
    clean : bool
        If *True*, apply ``clean_clinical_text`` to each record's text.
    """

    def __init__(
        self,
        filepath: str | Path,
        text_key: str = "text",
        label_key: str = "labels",
        clean: bool = False,
    ) -> None:
        self.filepath = Path(filepath)
        self.text_key = text_key
        self.label_key = label_key
        self.clean = clean

    def load_records(self) -> List[Dict[str, Any]]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records: List[Dict[str, Any]] = []
        for item in raw:
            text = item.get(self.text_key, "")
            if self.clean:
                text = clean_clinical_text(text)

            records.append({
                "text": text,
                "labels": item.get(self.label_key, []),
                "patient_id": item.get("patient_id",
                              item.get("metadata", {}).get("chunk_id", "")),
                "admission_id": item.get("admission_id",
                                item.get("metadata", {}).get("doc_id", "")),
            })

        return records

    def __repr__(self) -> str:
        return f"JSONDataPlugin(filepath={self.filepath})"


# ════════════════════════════════════════════════════════════════════
# Concrete: MIMIC-IV placeholder
# ════════════════════════════════════════════════════════════════════

class MIMICDataPlugin(BaseDataPlugin):
    """
    Placeholder data plugin for MIMIC-IV discharge summaries.

    .. note::

       This plugin will be implemented once PhysioNet access is granted.
       The expected workflow is:

       1. Read ``discharge.csv.gz`` from MIMIC-IV-Note
       2. Read ``diagnoses_icd.csv.gz`` from MIMIC-IV
       3. Merge on ``(subject_id, hadm_id)``
       4. Clean notes with ``clean_clinical_text``
       5. Return standardised records

    Parameters
    ----------
    mimic_dir : str | Path
        Root directory of the MIMIC-IV dataset.
    note_table : str
        Filename / table name for the notes (default: ``discharge``).
    diag_table : str
        Filename / table name for diagnoses (default: ``diagnoses_icd``).
    """

    def __init__(
        self,
        mimic_dir: str | Path,
        note_table: str = "discharge",
        diag_table: str = "diagnoses_icd",
    ) -> None:
        self.mimic_dir = Path(mimic_dir)
        self.note_table = note_table
        self.diag_table = diag_table

    def load_records(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "MIMICDataPlugin.load_records() is not yet implemented.  "
            "Provide PhysioNet-credentialed MIMIC-IV files in "
            f"'{self.mimic_dir}' and implement this method.  "
            "Expected tables: "
            f"'{self.note_table}', '{self.diag_table}'."
        )

    def __repr__(self) -> str:
        return f"MIMICDataPlugin(mimic_dir={self.mimic_dir})"
