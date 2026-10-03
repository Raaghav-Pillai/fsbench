"""Decide which documents appear, and where, under a filesystem condition.

All decisions use keyed random streams (see ``fsbench.rng``) so a sweep over one
knob changes only what that knob controls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from fsbench.config import FSConfig
from fsbench.distractors import generic_doc
from fsbench.docs import Doc
from fsbench.rng import keyed_rng, keyed_uniform
from fsbench.tasks import TaskInstance

JUNK_HOMES = [
    ("downloads",),
    ("desktop",),
    ("temp",),
    ("old", "misc"),
    ("backup", "2025"),
    ("shared", "to_sort"),
]

FILLER_DIRS = ["archive", "old", "misc", "backup", "2026", "shared", "working", "sorted", "keep", "z"]

GENERIC_DIRS = [
    "New folder", "New folder (2)", "stuff", "misc", "files", "untitled folder",
    "docs", "old", "temp2", "Archive", "share", "work", "folder1", "x",
]

COPY_SUFFIXES = [" (1)", "_copy", " - Copy", "_backup", " (2)"]

MAX_STEM = 80


@dataclass
class Entry:
    doc: Doc
    role: str  # required | trap | near | generic
    copy: int = 0
    fmt: str = ""
    dirs: tuple[str, ...] = ()
    name: str = ""
    scattered: bool = False
    noisy_name: bool = False

    @property
    def key(self) -> str:
        return f"{self.doc.doc_id}#{self.copy}"

    @property
    def path(self) -> str:
        return "/".join((*self.dirs, self.name))


def _noisy_stem(rng) -> str:
    n = rng.randint(1, 40)
    style = rng.randrange(14)
    return [
        lambda: str(rng.randint(1000, 999999)),
        lambda: "final",
        lambda: f"final{rng.randint(2, 4)}",
        lambda: "final_FINAL",
        lambda: f"final_v{rng.randint(2, 5)}_NEW({rng.randint(1, 9)})",
        lambda: f"scan_{rng.randint(1, 9999):04d}",
        lambda: f"Document{n}",
        lambda: f"untitled{n}",
        lambda: f"IMG_{rng.randint(1000, 9999)}",
        lambda: f"export ({n})",
        lambda: "Copy of report",
        lambda: f"download ({n})",
        lambda: f"{rng.getrandbits(32):08x}",
        lambda: f"notes_{n}",
    ][style]()


def _fit_depth(home: tuple[str, ...], depth: int | None, seed: int) -> tuple[str, ...]:
    if depth is None:
        return home
    if len(home) >= depth:
        return home[:depth]
    rng = keyed_rng(seed, "filler", "/".join(home))
    return home + tuple(rng.choice(FILLER_DIRS) for _ in range(depth - len(home)))


def _noisy_dirs(dirs: tuple[str, ...], cfg: FSConfig, seed: int) -> tuple[str, ...]:
    out = []
    for k in range(len(dirs)):
        prefix = "/".join(dirs[: k + 1])
        if keyed_uniform(seed, "dirnoise", prefix) < cfg.dirname_noise:
            out.append(keyed_rng(seed, "dirname", prefix).choice(GENERIC_DIRS))
        else:
            out.append(dirs[k])
    return tuple(out)


def select_entries(task: TaskInstance, cfg: FSConfig, world_seed: int, seed: int) -> list[Entry]:
    entries = [Entry(d, "required") for d in task.required]
    entries += [Entry(d, "trap") for d in task.inherent_traps]
    entries += [Entry(d, "trap") for d in task.stale_candidates
                if keyed_uniform(seed, "stale", d.doc_id) < cfg.stale_version_rate]

    near = task.near_pool()
    generic_i = 0
    seen_ids = {e.doc.doc_id for e in entries}
    for slot in range(cfg.distractors):
        if keyed_uniform(seed, "slot", slot) < cfg.semantic_similarity:
            doc = next(near)
            while doc.doc_id in seen_ids:
                doc = next(near)
            entries.append(Entry(doc, "near"))
        else:
            entries.append(Entry(generic_doc(world_seed, generic_i), "generic"))
            generic_i += 1
        seen_ids.add(entries[-1].doc.doc_id)

    for e in list(entries):
        if keyed_uniform(seed, "dup", e.doc.doc_id) < cfg.duplicate_rate:
            entries.append(Entry(e.doc, e.role, copy=1))
    return entries


def place_entries(entries: list[Entry], cfg: FSConfig, seed: int) -> None:
    for e in entries:
        doc = e.doc
        allowed = sorted(f for f in doc.formats if f in cfg.mime_types) or sorted(cfg.mime_types)
        e.fmt = keyed_rng(seed, "fmt", doc.doc_id).choice(allowed)

        home = doc.home
        if e.copy:
            if keyed_uniform(seed, "dupplace", doc.doc_id) < 0.5:
                home = keyed_rng(seed, "dupjunk", doc.doc_id).choice(JUNK_HOMES)
                e.scattered = True
        elif keyed_uniform(seed, "scatter", doc.doc_id) < cfg.scatter:
            home = keyed_rng(seed, "junk", doc.doc_id).choice(JUNK_HOMES)
            e.scattered = True
        e.dirs = _noisy_dirs(_fit_depth(home, cfg.depth, seed), cfg, seed)

        stem = doc.stem
        if e.copy:
            stem += keyed_rng(seed, "copysuffix", doc.doc_id).choice(COPY_SUFFIXES)
        if keyed_uniform(seed, "fname", doc.doc_id, e.copy) < cfg.filename_noise:
            stem = _noisy_stem(keyed_rng(seed, "fnamechoice", doc.doc_id, e.copy))
            e.noisy_name = True
        e.name = f"{stem[:MAX_STEM]}.{e.fmt}"

    _canonicalize_dirs(entries, seed)
    _dedupe_names(entries, seed)
    if cfg.max_entries_per_dir:
        _split_overfull(entries, cfg.max_entries_per_dir)
        _dedupe_names(entries, seed)


def _neutral_order(entries: list[Entry], seed: int) -> list[Entry]:
    # Role must not influence who keeps the un-suffixed name.
    return sorted(entries, key=lambda e: keyed_uniform(seed, "order", e.key))


def _canonicalize_dirs(entries: list[Entry], seed: int) -> None:
    """Merge folders whose names differ only by case, as case-insensitive filesystems would."""
    spelling: dict[tuple[str, ...], str] = {}
    for e in _neutral_order(entries, seed):
        out: list[str] = []
        for seg in e.dirs:
            key = (*(s.casefold() for s in out), seg.casefold())
            out.append(spelling.setdefault(key, seg))
        e.dirs = tuple(out)


def _dedupe_names(entries: list[Entry], seed: int) -> None:
    taken: set[str] = set()
    for e in _neutral_order(entries, seed):
        stem, ext = e.name.rsplit(".", 1)
        name, k = e.name, 2
        while "/".join((*e.dirs, name)).casefold() in taken:
            name = f"{stem} ({k}).{ext}"
            k += 1
        e.name = name
        taken.add(e.path.casefold())


def _split_overfull(entries: list[Entry], limit: int) -> None:
    files_in: dict[tuple[str, ...], list[Entry]] = defaultdict(list)
    subdirs: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for e in entries:
        files_in[e.dirs].append(e)
        for k in range(len(e.dirs)):
            subdirs[e.dirs[:k]].add(e.dirs[k].casefold())
    for d, es in files_in.items():
        if len(es) + len(subdirs[d]) <= limit:
            continue
        es.sort(key=lambda e: e.name.casefold())
        for i in range(0, len(es), limit):
            for e in es[i : i + limit]:
                e.dirs = (*d, f"part_{i // limit + 1:02d}")
