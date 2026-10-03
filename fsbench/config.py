"""Filesystem condition: the independent variables of the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

from fsbench.docs import ALL_FORMATS, mime_types_for_diversity


@dataclass(frozen=True)
class FSConfig:
    """One filesystem condition. Every knob defaults to its 'easy' value.

    depth               Directory levels above every file. None keeps each document's
                        natural semantic depth; smaller values truncate the semantic path,
                        larger values pad it with filler folders (archive/, misc/, ...).
    filename_noise      Probability a file loses its descriptive name (final_v2_NEW(3).pdf).
    dirname_noise       Probability a folder loses its descriptive name (New folder/, stuff/).
    scatter             Probability a file lands in a junk location (downloads/, desktop/, temp/).
    distractors         Number of irrelevant documents added.
    semantic_similarity Fraction of distractors drawn from the task's near-miss pool
                        (other clients' invoices, look-alike names) rather than unrelated docs.
    duplicate_rate      Probability any file also appears as an exact copy elsewhere.
    stale_version_rate  Probability each outdated version of a required doc is present.
    mime_types          File formats documents may be rendered as.
    max_entries_per_dir If set, overfull folders are split into part_NN/ subfolders.
    """

    depth: int | None = None
    filename_noise: float = 0.0
    dirname_noise: float = 0.0
    scatter: float = 0.0
    distractors: int = 0
    semantic_similarity: float = 0.5
    duplicate_rate: float = 0.0
    stale_version_rate: float = 0.0
    mime_types: tuple[str, ...] = ("txt",)
    max_entries_per_dir: int | None = None

    def __post_init__(self) -> None:
        for name in ("filename_noise", "dirname_noise", "scatter", "semantic_similarity",
                     "duplicate_rate", "stale_version_rate"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.depth is not None and self.depth < 0:
            raise ValueError("depth must be >= 0")
        if self.distractors < 0:
            raise ValueError("distractors must be >= 0")
        if self.max_entries_per_dir is not None and self.max_entries_per_dir < 2:
            raise ValueError("max_entries_per_dir must be >= 2")
        if not self.mime_types:
            raise ValueError("mime_types must not be empty")
        bad = [m for m in self.mime_types if m not in ALL_FORMATS]
        if bad:
            raise ValueError(f"unsupported mime types {bad}; choose from {ALL_FORMATS}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mime_types"] = list(self.mime_types)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FSConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"mime_diversity"}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d = dict(d)
        if "mime_diversity" in d:
            d["mime_types"] = mime_types_for_diversity(int(d.pop("mime_diversity")))
        if "mime_types" in d:
            d["mime_types"] = tuple(d["mime_types"])
        return cls(**d)

    def with_value(self, key: str, value) -> "FSConfig":
        if key == "mime_diversity":
            return replace(self, mime_types=mime_types_for_diversity(int(value)))
        return replace(self, **{key: value})


PRESETS: dict[str, FSConfig] = {
    "organized": FSConfig(mime_types=("txt", "csv", "pdf")),
    "moderate": FSConfig(
        depth=5,
        filename_noise=0.3,
        dirname_noise=0.2,
        scatter=0.2,
        distractors=60,
        semantic_similarity=0.4,
        duplicate_rate=0.1,
        stale_version_rate=0.3,
        mime_types=("txt", "csv", "pdf", "xlsx", "docx"),
    ),
    "horrible": FSConfig(
        depth=8,
        filename_noise=0.8,
        dirname_noise=0.6,
        scatter=0.5,
        distractors=300,
        semantic_similarity=0.7,
        duplicate_rate=0.3,
        stale_version_rate=1.0,
        mime_types=ALL_FORMATS,
        max_entries_per_dir=40,
    ),
}
