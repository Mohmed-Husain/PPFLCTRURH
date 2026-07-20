"""
Text preprocessing utilities.

**Kept verbatim** from Phase 1 notebook (``01_Dataset_Generation.ipynb``):
  - ``clean_text``          – unicode normalisation, whitespace collapse
  - ``split_into_paragraphs`` – double-newline splitting
  - ``chunk_paragraphs``    – paragraph-aware sliding-window chunker
  - ``word_count``

**Added** for clinical text:
  - ``clean_clinical_text`` – extends ``clean_text`` with MIMIC-style
    de-identification pattern removal and section-header normalisation.
"""

from __future__ import annotations

import re
from typing import List


# ─────────────────────────────────────────────────────────────────────
# Kept verbatim from Phase 1 notebook
# ─────────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    """Count words in *text*."""
    return len(text.split())


def clean_text(text: str) -> str:
    """
    Clean extracted text: normalise whitespace, remove control characters,
    and fix common encoding artefacts.

    Kept from Phase 1 notebook ``clean_text()``.
    """
    if not text:
        return ""
    # Remove form feeds and null bytes
    text = text.replace("\x0c", " ").replace("\x00", "")
    # Normalise various dash types to standard hyphen
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)
    # Normalise quotes
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    # Collapse multiple whitespace (but preserve paragraph breaks)
    text = re.sub(r"[^\S\n]+", " ", text)       # spaces/tabs to single space
    text = re.sub(r"\n{3,}", "\n\n", text)       # 3+ newlines to 2
    # Remove lines that are just whitespace
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split *text* into paragraphs by double-newline boundaries.
    Returns non-empty paragraph strings.

    Kept from Phase 1 notebook ``split_into_paragraphs()``.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


def chunk_paragraphs(
    paragraphs: List[str],
    target_words: int = 512,
    overlap_words: int = 64,
    min_words: int = 50,
) -> List[str]:
    """
    Group paragraphs into chunks targeting *target_words* word count.
    Preserves paragraph boundaries.  Uses word-count-based overlap.

    Kept from Phase 1 notebook ``chunk_paragraphs()``.
    """
    chunks: List[str] = []
    current_chunk_paras: List[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_wc = word_count(para)

        # If adding this paragraph exceeds the target and we have content
        if current_word_count + para_wc > target_words and current_chunk_paras:
            # Save current chunk
            chunk_text = "\n\n".join(current_chunk_paras)
            if word_count(chunk_text) >= min_words:
                chunks.append(chunk_text)

            # Start new chunk with overlap
            overlap_paras: List[str] = []
            overlap_wc = 0
            for p in reversed(current_chunk_paras):
                p_wc = word_count(p)
                if overlap_wc + p_wc > overlap_words and overlap_paras:
                    break
                overlap_paras.insert(0, p)
                overlap_wc += p_wc

            current_chunk_paras = overlap_paras
            current_word_count = overlap_wc

        current_chunk_paras.append(para)
        current_word_count += para_wc

    # Final chunk
    if current_chunk_paras:
        chunk_text = "\n\n".join(current_chunk_paras)
        if word_count(chunk_text) >= min_words:
            chunks.append(chunk_text)

    return chunks


# ─────────────────────────────────────────────────────────────────────
# Added for clinical text (MIMIC compatibility)
# ─────────────────────────────────────────────────────────────────────

# MIMIC de-identification pattern: [**First Name**], [**2100-01-01**], etc.
_MIMIC_DEIDENT_RE = re.compile(r"\[\*\*[^\]]*\*\*\]")

# Common clinical section headers (normalise casing)
_SECTION_HEADER_RE = re.compile(
    r"^(HISTORY OF PRESENT ILLNESS|DISCHARGE DIAGNOSIS|"
    r"DISCHARGE MEDICATIONS|HOSPITAL COURSE|"
    r"CHIEF COMPLAINT|PAST MEDICAL HISTORY|"
    r"SOCIAL HISTORY|FAMILY HISTORY|"
    r"PHYSICAL EXAMINATION|LABORATORY DATA|"
    r"DISCHARGE INSTRUCTIONS|DISCHARGE CONDITION|"
    r"ADMISSION DATE|DISCHARGE DATE|"
    r"ALLERGIES|MEDICATIONS ON ADMISSION)\s*:?\s*",
    re.IGNORECASE | re.MULTILINE,
)


def clean_clinical_text(text: str, *, remove_deident: bool = True) -> str:
    """
    Extend ``clean_text`` with clinical-note-specific cleaning:

    1. Remove MIMIC-style de-identification brackets ``[**...**]``.
    2. Normalise common clinical section headers.
    3. Apply all standard ``clean_text`` transformations.

    Parameters
    ----------
    text : str
        Raw clinical note.
    remove_deident : bool
        If *True*, replace ``[**...**]`` placeholders with a single space.
    """
    if not text:
        return ""

    if remove_deident:
        text = _MIMIC_DEIDENT_RE.sub(" ", text)

    # Normalise section headers to Title Case with colon
    def _normalise_header(m: re.Match) -> str:
        return m.group(1).strip().title() + ":\n"

    text = _SECTION_HEADER_RE.sub(_normalise_header, text)

    # Delegate the rest to the original clean_text
    return clean_text(text)


def chunk_clinical_text(
    text: str,
    target_words: int = 512,
    overlap_words: int = 64,
    min_words: int = 50,
) -> List[str]:
    """
    Convenience: clean → split paragraphs → chunk.

    Parameters
    ----------
    text : str
        Raw or lightly cleaned clinical note.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    cleaned = clean_clinical_text(text)
    paragraphs = split_into_paragraphs(cleaned)
    return chunk_paragraphs(paragraphs, target_words, overlap_words, min_words)
