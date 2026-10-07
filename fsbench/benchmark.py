"""Benchmark identity, task capabilities, and evaluator-only manifest storage."""
import json
import os
import re
from pathlib import Path

from fsbench.paths import long_path

SYNTHETIC_VERSION = "synthetic-v2"
CAPABILITIES = {
    "retrieve": ["retrieval"],
    "conflict": ["source_selection", "temporal_reasoning"],
    "reconcile": ["multi_file_reasoning", "reconciliation"],
    "workflow": ["workflow_execution", "reconciliation"],
}


def load_manifest(env):
    env = long_path(env)
    local = env / "manifest.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    ref = json.loads((env / "evaluation_ref.json").read_text(encoding="utf-8"))
    identifier = ref["instance_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("invalid evaluator reference")
    configured = os.environ.get("FSBENCH_EVALUATOR_ROOT")
    roots = [long_path(configured)] if configured else [p / ".private" for p in env.parents]
    for root in roots:
        path = root / "manifests" / (identifier + ".json")
        if path.is_file():
            m = json.loads(path.read_text(encoding="utf-8"))
            if m["env_id"] != identifier or m["benchmark_version"] != ref["benchmark_version"]:
                raise ValueError("evaluator reference mismatch")
            return m
    raise ValueError("private evaluator manifest unavailable; configure FSBENCH_EVALUATOR_ROOT")


def identity(m):
    return {"benchmark_family": m.get("benchmark_family", "synthetic"),
            "benchmark_version": m.get("benchmark_version", "synthetic-legacy"),
            "task_families": m.get("task_families", CAPABILITIES.get(m["task"]["type"], [])),
            "workspace_domain": m.get("workspace_domain"), "split": m.get("split", "dev"),
            "world_id": m.get("world_id", str(m["seeds"]["world"])),
            "difficulty_level": m.get("difficulty_level"),
            "task_template": m.get("task_template", m["task"]["type"])}


def require_evaluator(m, enabled=False):
    if m.get("split") in ("validation", "test") and not enabled:
        raise ValueError("held-out answers and labels require the evaluator workflow (--evaluator)")
