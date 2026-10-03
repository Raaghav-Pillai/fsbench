"""Score a run against the hidden manifest and derive file-I/O metrics from its trace."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fsbench.paths import display_path, long_path
from fsbench.tasks import TASK_TYPES


def parse_answer(raw) -> dict:
    """Accept a dict, a JSON string, or free text containing a JSON object."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    candidates = fenced + re.findall(r"\{.*\}", raw, flags=re.S)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_run(run_dir: str | Path) -> dict:
    run = long_path(run_dir)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    manifest = json.loads((long_path(meta["env_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    answer = json.loads((run / "answer.json").read_text(encoding="utf-8"))
    events = load_trace(run / "trace.jsonl")

    task_type = manifest["task"]["type"]
    outcome = TASK_TYPES[task_type].score(answer, manifest["ground_truth"], run / "workspace")
    return {
        "env_id": manifest["env_id"],
        "task_type": task_type,
        "task_level": manifest["task"]["level"],
        "agent": meta["agent"],
        "toolset": meta["toolset"],
        "allow_writes": meta["allow_writes"],
        "success": bool(outcome["success"]),
        "score": round(float(outcome["score"]), 4),
        "outcome": outcome,
        **trace_metrics(events, manifest, success=bool(outcome["success"])),
        "wall_time_s": meta.get("wall_time_s"),
        "agent_error": bool(meta.get("agent_error")),
        "usage": meta.get("usage"),
    }


def trace_metrics(events: list[dict], manifest: dict, *, success: bool) -> dict:
    files = {f["path"]: f for f in manifest["files"]}
    required = set(manifest["required_doc_ids"])

    read_paths = list(dict.fromkeys(p for e in events for p in e["read"]))
    seen_paths = list(dict.fromkeys([p for e in events for p in e["seen"]] + read_paths))
    known_read = [p for p in read_paths if p in files]

    def docs(paths):
        return {files[p]["doc_id"] for p in paths if p in files}

    read_docs, seen_docs = docs(read_paths), docs(seen_paths)
    useful = [p for p in known_read if files[p]["doc_id"] in required]
    roles = Counter(files[p]["role"] for p in known_read)
    cats = Counter(e["category"] for e in events)
    read_calls = [e for e in events if e["category"] == "read" and e["ok"]]
    path_calls = [e for e in events if e["path_args"]]
    missing_calls = [e for e in events if e["missing"]]
    first_required = next((e["step"] for e in events if any(
        p in files and files[p]["doc_id"] in required for p in e["read"])), None)
    all_required_step = None
    got: set[str] = set()
    for e in events:
        got |= {files[p]["doc_id"] for p in e["read"] if p in files} & required
        if got == required:
            all_required_step = e["step"]
            break

    oracle = manifest["oracle"]
    n_calls = len(events)
    exposed = roles["trap"] + roles["near"] > 0
    output_chars = sum(e["output_chars"] for e in events)
    return {
        "n_calls": n_calls,
        "n_navigate": cats["navigate"],
        "n_search": cats["search"],
        "n_read": cats["read"],
        "n_write": cats["write"],
        "n_errors": sum(1 for e in events if not e["ok"]),
        "files_seen": len(seen_paths),
        "files_read": len(read_paths),
        "redundant_reads": max(0, len(read_calls) - len(read_paths)),
        "required_recall_read": round(len(read_docs & required) / len(required), 4),
        "required_recall_seen": round(len(seen_docs & required) / len(required), 4),
        "read_precision": round(len(useful) / len(read_paths), 4) if read_paths else None,
        "trap_reads": roles["trap"],
        "near_distractor_reads": roles["near"],
        "generic_distractor_reads": roles["generic"],
        "trap_exposed": exposed,
        "recovered": exposed and success,
        "path_hallucination_rate": round(len(missing_calls) / len(path_calls), 4) if path_calls else 0.0,
        "hallucinated_paths": sorted({p for e in missing_calls for p in e["missing"]}),
        "chars_read": sum(e["output_chars"] for e in read_calls),
        "bytes_read_on_disk": sum(files[p]["bytes"] for p in known_read),
        "tool_output_chars": output_chars,
        "est_tool_output_tokens": output_chars // 4,
        "truncated_outputs": sum(1 for e in events if e["truncated"]),
        "steps_to_first_required": first_required,
        "steps_to_all_required": all_required_step,
        "optimal_calls": oracle["optimal_calls"],
        "browse_lower_bound": oracle["browse_lower_bound"],
        "navigation_regret": round(n_calls / oracle["optimal_calls"], 4) if n_calls else None,
        "browse_regret": round(n_calls / oracle["browse_lower_bound"], 4) if n_calls else None,
        "agent_written_files": sorted({p for e in events for p in e["written"]}),
    }


def find_runs(root: str | Path) -> list[Path]:
    return sorted(Path(display_path(p.parent)) for p in long_path(root).rglob("meta.json")
                  if (p.parent / "answer.json").exists())


def find_envs(root: str | Path) -> list[Path]:
    lroot = long_path(root)
    if (lroot / "manifest.json").exists():
        return [Path(display_path(lroot))]
    return sorted(Path(display_path(p.parent)) for p in lroot.rglob("manifest.json")
                  if (p.parent / "workspace").is_dir())


def flatten_row(metrics: dict, manifest: dict) -> dict:
    row = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    row["fields_correct"] = json.dumps(metrics["outcome"].get("fields", {}))
    for k, v in (metrics.get("usage") or {}).items():
        row[f"usage.{k}"] = v
    for k, v in manifest["config"].items():
        row[f"cfg.{k}"] = "+".join(v) if isinstance(v, list) else v
    for k, v in manifest["stats"].items():
        if not isinstance(v, dict):
            row[f"stat.{k}"] = v
    sweep = manifest.get("sweep") or {}
    row["sweep.vars"] = ";".join(f"{k}={v}" for k, v in sweep.get("vars", {}).items())
    row["sweep.replicate"] = sweep.get("replicate")
    for k, v in manifest["seeds"].items():
        row[f"seed.{k}"] = v
    return row


def evaluate_runs(root: str | Path, out_csv: str | Path | None = None) -> list[dict]:
    rows = []
    for run in find_runs(root):
        metrics = evaluate_run(run)
        (long_path(run) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        meta = json.loads((long_path(run) / "meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((long_path(meta["env_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        rows.append({"run_dir": str(run), **flatten_row(metrics, manifest)})
    if out_csv and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


SUMMARY_METRICS = [
    "success", "score", "required_recall_read", "read_precision", "path_hallucination_rate",
    "n_calls", "files_read", "navigation_regret", "est_tool_output_tokens",
]


def summarize(rows: list[dict], by: tuple[str, ...] = ("agent", "toolset", "task_type", "sweep.vars")) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        s = dict(zip(by, key))
        s["n"] = len(rs)
        for m in SUMMARY_METRICS:
            vals = [float(r[m]) for r in rs if r.get(m) is not None]
            s[m] = round(sum(vals) / len(vals), 4) if vals else None
        trapped = [r for r in rs if r.get("trap_exposed")]
        s["recovery_rate"] = round(sum(r["recovered"] for r in trapped) / len(trapped), 4) if trapped else None
        out.append(s)
    return out
