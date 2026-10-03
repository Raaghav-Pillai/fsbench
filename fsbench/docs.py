"""Format-independent document model.

A ``Doc`` holds structured content (title, key/value fields, an optional table,
free-text paragraphs) plus its *semantic* location (``home``) and clean filename
(``stem``). The layout stage decides the actual path, name, and file format, so
the same Doc can be rendered identically into any filesystem condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALL_FORMATS = ("txt", "md", "csv", "json", "pdf", "xlsx", "docx")

# Order used when MIME diversity is given as a count (1 = txt only, 7 = all).
DIVERSITY_ORDER = ("txt", "pdf", "csv", "xlsx", "docx", "json", "md")

TABULAR_FORMATS = ("csv", "xlsx", "json", "txt", "md", "pdf", "docx")
FORM_FORMATS = ("pdf", "docx", "txt", "md", "json", "xlsx")
PROSE_FORMATS = ("pdf", "docx", "txt", "md")


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]


@dataclass
class Doc:
    doc_id: str
    kind: str
    title: str
    stem: str
    home: tuple[str, ...]
    fields: list[tuple[str, str]] = field(default_factory=list)
    table: Table | None = None
    paragraphs: list[str] = field(default_factory=list)
    formats: tuple[str, ...] = ALL_FORMATS


def mime_types_for_diversity(n: int) -> tuple[str, ...]:
    if not 1 <= n <= len(DIVERSITY_ORDER):
        raise ValueError(f"MIME diversity must be between 1 and {len(DIVERSITY_ORDER)}")
    return DIVERSITY_ORDER[:n]


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def plain_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"
