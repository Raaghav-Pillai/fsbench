"""Reproducible trial schedules and checks on the agent-visible input."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from fsbench.paths import long_path
from fsbench.benchmark import load_manifest, identity as benchmark_identity
from fsbench.rng import keyed_rng
from fsbench.tools import IndexedTools


def workspace_inventory(root: str | Path) -> dict[str, str]:
    root = long_path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"workspace links are not supported: {path}")
        rel = path.relative_to(root).as_posix()
        result[rel + "/" if path.is_dir() else rel] = (
            "directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_environment(env: str | Path, *, check_index: bool = False) -> dict:
    env = long_path(env)
    m = load_manifest(env)
    task = json.loads((env / "task.json").read_text(encoding="utf-8"))
    if task != {k: m["task"][k] for k in ("task_id", "prompt", "answer_schema")}:
        raise ValueError(f"agent task does not match manifest: {env}")
    inventory = workspace_inventory(env / "workspace")
    actual = {p: digest for p, digest in inventory.items() if not p.endswith("/")}
    declared = {f["path"]: f for f in m["files"]}
    if actual.keys() != declared.keys():
        raise ValueError(f"workspace does not match manifest file inventory: {env}")
    for p, digest in actual.items():
        if declared[p].get("sha256") not in (None, digest):
            raise ValueError(f"file bytes changed: {env / 'workspace' / p}")
        if declared[p].get("modified_at"):
            from datetime import datetime
            expected = datetime.fromisoformat(declared[p]["modified_at"]).timestamp()
            if abs((env / "workspace" / p).stat().st_mtime - expected) > 1:
                raise ValueError(f"file modification time changed: {p}")
    if check_index:
        tools = IndexedTools(env / "workspace")
        index = tools._build_index()
        if set(index["docs"]) != set(actual):
            raise ValueError(f"index file inventory mismatch: {env}")
        for p, (_, _, text) in index["docs"].items():
            if not text.strip():
                raise ValueError(f"empty extracted index document: {p}")
    return {"workspace_sha256": fingerprint(inventory),
            "input_sha256": fingerprint({"workspace": inventory,
                "manifest": m, "task": task})}


def build_schedule(envs: list[Path], toolsets: list[str], *, order_seed: int = 7,
                   check_index: bool = False, arms: list[tuple[str, str]] | None = None,
                   models: list[str] | None = None, trials_per_cell: int = 1) -> list[dict]:
    """Shuffle noise x toolset cells inside each seed, independently of discovery order."""
    groups = defaultdict(list)
    if trials_per_cell < 1:
        raise ValueError("trials_per_cell must be positive")
    manifests = [load_manifest(env) for env in envs]
    if manifests and all(m.get("benchmark_family", "synthetic") == "synthetic" and all("document_sha256" in f for f in m["files"]) for m in manifests):
        from fsbench.difficulty import verify_composed
        verify_composed(envs)
    seen = set()
    for env in sorted(envs):
        m = load_manifest(env)
        seeds = m["seeds"]
        key = (m["task"]["type"], seeds["world"], seeds["task"], seeds["layout"])
        cell = (key, m["config"]["filename_noise"], (m.get("condition") or {}).get("config_hash"))
        if cell in seen:
            raise ValueError(f"duplicate environment condition: {env}")
        seen.add(cell)
        identity = validate_environment(env, check_index=check_index)
        for ts, policy in sorted(set(arms or [(t, "naturalistic") for t in toolsets])):
            for model in sorted(set(models or [""])):
                for trial in range(1, trials_per_cell + 1):
                    groups[key].append({"env_dir": str(env.resolve()), "env_id": m["env_id"],
                        **benchmark_identity(m),
                        "tool_policy": policy, "condition_id": fingerprint(m["config"]),
                        "condition_label": (m.get("condition") or {}).get("label"),
                        "model": model, "trial_index": trial,
                        "toolset": ts, "task": key[0], "world_seed": seeds["world"],
                        "task_seed": seeds["task"], "layout_seed": seeds["layout"],
                        "filename_noise": m["config"]["filename_noise"], **identity})
    schedule = []
    for key in sorted(groups):
        cells = sorted(groups[key], key=lambda c: (c["condition_id"], c["toolset"], c["tool_policy"],
                                                   c["model"], c["trial_index"], c["env_id"]))
        keyed_rng(order_seed, "experiment-order-v1", *key).shuffle(cells)
        schedule.extend(cells)
    return [{**cell, "execution_order": i, "experiment_order_seed": order_seed}
            for i, cell in enumerate(schedule, 1)]


def save_plan(path: Path, plan: dict) -> None:
    """A resume must preserve the schedule, settings and original input fingerprints."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError(f"experiment plan differs from {path}; use a new output directory")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
