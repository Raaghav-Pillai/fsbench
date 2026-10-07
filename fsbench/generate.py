"""Build one benchmark environment on disk.

Output layout::

    <out_dir>/
        workspace/      the only directory an agent may see
        task.json       agent-visible: prompt and answer schema
        manifest.json   hidden: ground truth, file roles, oracle costs, structure stats
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from fsbench.config import FSConfig
from fsbench.benchmark import CAPABILITIES, SYNTHETIC_VERSION
from fsbench.layout import FILENAME_NOISE_VERSION, Entry, place_entries, select_entries
from fsbench.paths import long_path
from fsbench.render import render
from fsbench.tasks import TASK_TYPES, extract_decoy
from fsbench.world import build_world

MANIFEST_VERSION = 2


def generate_env(
    task_type: str,
    cfg: FSConfig,
    out_dir: str | Path,
    *,
    world_seed: int = 0,
    task_seed: int = 0,
    layout_seed: int = 0,
    env_id: str | None = None,
    sweep: dict | None = None,
    label: str | None = None,
) -> dict:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task type {task_type!r}; choose from {sorted(TASK_TYPES)}")
    world = build_world(world_seed)
    task = TASK_TYPES[task_type].build(world, task_seed)
    entries = select_entries(task, cfg, world_seed, layout_seed)
    place_entries(entries, cfg, layout_seed, world_seed=world_seed)

    out = long_path(out_dir)
    workspace = out / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    rendered: dict[tuple[str, str], bytes] = {}
    files = []
    for e in sorted(entries, key=lambda e: e.path):
        data = rendered.get((e.doc.doc_id, e.fmt))
        if data is None:
            data = rendered[(e.doc.doc_id, e.fmt)] = render(e.doc, e.fmt)
        target = workspace.joinpath(*e.dirs, e.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({
            "path": e.path,
            "doc_id": e.doc.doc_id,
            "role": e.role,
            "kind": e.doc.kind,
            "format": e.fmt,
            "bytes": len(data),
            "copy": e.copy,
            "scattered": e.scattered,
            "noisy_name": e.noisy_name,
            "clean_name": e.clean_name,
            "filename_noise_score": e.noise_score,
            "sha256": hashlib.sha256(data).hexdigest(),
            "document_sha256": hashlib.sha256(json.dumps(asdict(e.doc), sort_keys=True).encode()).hexdigest(),
        })

    required_ids = [d.doc_id for d in task.required]
    trap_ids = sorted({e.doc.doc_id for e in entries if e.role == "trap"})
    env_id = env_id or Path(out_dir).name
    present = {f["doc_id"] for f in files}
    decoys = {k: v for k, v in task.decoy_answers.items() if k in present}
    by_id = {e.doc.doc_id: e.doc for e in entries}
    for f in files:
        if f["role"] in ("trap", "near") and f["doc_id"] not in decoys:
            extra = extract_decoy(task.task_type, by_id[f["doc_id"]])
            if extra:
                decoys[f["doc_id"]] = extra
    manifest = {
        "benchmark_family": "synthetic",
        "benchmark_version": SYNTHETIC_VERSION,
        "task_families": CAPABILITIES[task_type],
        "manifest_version": MANIFEST_VERSION,
        "filename_noise_version": FILENAME_NOISE_VERSION,
        "env_id": env_id,
        "task": {
            "type": task.task_type,
            "level": task.level,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "answer_schema": task.answer_schema,
            "output_path": task.output_path,
        },
        "ground_truth": task.ground_truth,
        "seeds": {"world": world_seed, "task": task_seed, "layout": layout_seed},
        "config": cfg.to_dict(),
        "condition": _condition(cfg, label, sweep),
        "sweep": sweep,
        "required_doc_ids": required_ids,
        "trap_doc_ids": trap_ids,
        "decoy_answers": decoys,
        "oracle": _oracle_costs(entries, required_ids, task.output_path),
        "stats": _structure_stats(entries, files),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "task.json").write_text(
        json.dumps({"task_id": task.task_id, "prompt": task.prompt, "answer_schema": task.answer_schema}, indent=2),
        encoding="utf-8",
    )
    return manifest


def _condition(cfg: FSConfig, label: str | None, sweep: dict | None) -> dict:
    if sweep and sweep.get("vars"):
        cond_label = ";".join(f"{k}={v}" for k, v in sweep["vars"].items())
    else:
        cond_label = label or f"filename_noise={cfg.filename_noise:g}"
    dumped = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return {"label": cond_label, "config_hash": hashlib.sha1(dumped.encode()).hexdigest()[:8]}


def _oracle_costs(entries: list[Entry], required_ids: list[str], output_path: str | None) -> dict:
    """Lower bounds on tool calls.

    known_path_optimal_calls  a path-omniscient agent: one read per required doc (+ one write).
    discovery_optimal_calls   an agent that knows what it needs but must find it by listing
                              directories from the root: every ancestor folder of the shallowest
                              copy of each required doc is listed once, then each doc is read.

    Toolsets with search can beat discovery_optimal_calls (discovery regret < 1). That is
    expected and measures how much search helps.
    """
    writes = 1 if output_path else 0
    listed: set[tuple[str, ...]] = set()
    for doc_id in required_ids:
        copies = [e for e in entries if e.doc.doc_id == doc_id]
        best = min(copies, key=lambda e: (len(e.dirs), e.path))
        for k in range(len(best.dirs) + 1):
            listed.add(tuple(best.dirs[:k]))
    reads = len(required_ids)
    return {
        "min_reads": reads,
        "min_writes": writes,
        "known_path_optimal_calls": reads + writes,
        "discovery_optimal_calls": len(listed) + reads + writes,
    }


def _structure_stats(entries: list[Entry], files: list[dict]) -> dict:
    dirs: set[tuple[str, ...]] = {()}
    children: dict[tuple[str, ...], set[str]] = {}
    for e in entries:
        for k in range(len(e.dirs) + 1):
            dirs.add(tuple(e.dirs[:k]))
        for k in range(len(e.dirs)):
            children.setdefault(tuple(e.dirs[:k]), set()).add(e.dirs[k].casefold())
        children.setdefault(e.dirs, set()).add(e.name.casefold())
    depths = [len(e.dirs) for e in entries]
    formats = Counter(f["format"] for f in files)
    n = len(files)
    entropy = -sum(c / n * math.log2(c / n) for c in formats.values()) if n else 0.0
    roles = Counter(f["role"] for f in files)
    return {
        "n_files": n,
        "n_dirs": len(dirs),
        "max_depth": max(depths, default=0),
        "mean_depth": round(sum(depths) / n, 3) if n else 0.0,
        "max_entries_per_dir": max((len(c) for c in children.values()), default=0),
        "format_counts": dict(sorted(formats.items())),
        "format_entropy_bits": round(entropy, 4),
        "frac_noisy_names": round(sum(f["noisy_name"] for f in files) / n, 4) if n else 0.0,
        "frac_scattered": round(sum(f["scattered"] for f in files) / n, 4) if n else 0.0,
        "role_counts": dict(sorted(roles.items())),
        "duplicate_files": sum(1 for f in files if f["copy"]),
        "total_bytes": sum(f["bytes"] for f in files),
    }
