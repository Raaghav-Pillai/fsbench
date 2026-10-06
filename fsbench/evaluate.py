"""Score a run against the hidden manifest and derive file-I/O metrics from its trace."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from fsbench.paths import display_path, long_path
from fsbench.tasks import TASK_TYPES, _money_eq, _norm, parse_date, parse_money

FAILURE_TYPES = [
    "timeout",
    "agent_error",
    "write_failure",
    "output_format_failure",
    "retrieval_failure",
    "stale_version_confusion",
    "distractor_confusion",
    "reasoning_failure",
]

SUMMARY_METRICS = [
    "success", "evidence_found", "n_calls", "files_read", "discovery_regret",
    "total_tokens", "tool_output_tokens", "cost_usd", "wall_time_s", "llm_time_s",
]


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
    io = trace_metrics(events, manifest)
    usage = meta.get("usage") or {}
    timing = _timing(events, meta, usage)
    tokens = _tokens(usage, io["tool_output_chars"])
    failure = classify_failure(outcome, io, answer, meta, manifest, run / "workspace")
    return {
        "env_id": manifest["env_id"],
        "task_type": task_type,
        "task_level": manifest["task"]["level"],
        "agent": meta["agent"],
        "model": meta.get("model") or (meta.get("agent_describe") or {}).get("model"),
        "toolset": meta["toolset"],
        "allow_writes": meta["allow_writes"],
        "success": bool(outcome["success"]),
        "score": round(float(outcome["score"]), 4),
        "answer_correct": bool(outcome["success"]),
        "evidence_found": io["required_recall_read"] == 1.0,
        "evidence_seen": io["required_recall_seen"] == 1.0,
        "failure_type": failure["failure_type"],
        "failure_detail": failure["failure_detail"],
        "outcome": outcome,
        **io,
        **timing,
        **tokens,
        "agent_error": bool(meta.get("agent_error")),
        "usage": usage,
        "condition": (manifest.get("condition") or {}).get("label"),
        "config_hash": (manifest.get("condition") or {}).get("config_hash"),
        "fsbench_version": meta.get("fsbench_version"),
        "git_commit": meta.get("git_commit"),
        "agent_prompt_version": (meta.get("agent_describe") or {}).get("agent_prompt_version"),
    }


def _oracle_calls(manifest: dict) -> tuple[int, int]:
    oracle = manifest["oracle"]
    known = oracle.get("known_path_optimal_calls") or oracle.get("optimal_calls")
    discovery = oracle.get("discovery_optimal_calls") or oracle.get("browse_lower_bound")
    return known, discovery


def trace_metrics(events: list[dict], manifest: dict) -> dict:
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

    known, discovery = _oracle_calls(manifest)
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
        "read_doc_ids": sorted(read_docs),
        "redundant_reads": max(0, len(read_calls) - len(read_paths)),
        "required_recall_read": round(len(read_docs & required) / len(required), 4),
        "required_recall_seen": round(len(seen_docs & required) / len(required), 4),
        "read_precision": round(len(useful) / len(read_paths), 4) if read_paths else None,
        "trap_reads": roles["trap"],
        "near_distractor_reads": roles["near"],
        "generic_distractor_reads": roles["generic"],
        "trap_exposed": exposed,
        "recovered": exposed and False,  # filled after we know success
        "path_hallucination_rate": round(len(missing_calls) / len(path_calls), 4) if path_calls else 0.0,
        "hallucinated_paths": sorted({p for e in missing_calls for p in e["missing"]}),
        "chars_read": sum(e["output_chars"] for e in read_calls),
        "bytes_read_on_disk": sum(files[p]["bytes"] for p in known_read),
        "tool_output_chars": output_chars,
        "tool_output_tokens": output_chars // 4,
        "truncated_outputs": sum(1 for e in events if e["truncated"]),
        "steps_to_first_required": first_required,
        "steps_to_all_required": all_required_step,
        "known_path_optimal_calls": known,
        "discovery_optimal_calls": discovery,
        "known_path_regret": round(n_calls / known, 4) if n_calls and known else None,
        "discovery_regret": round(n_calls / discovery, 4) if n_calls and discovery else None,
        "agent_written_files": sorted({p for e in events for p in e["written"]}),
    }


def _timing(events: list[dict], meta: dict, usage: dict) -> dict:
    by_cat: dict[str, float] = defaultdict(float)
    parse_ms = 0.0
    for e in events:
        by_cat[e["category"]] += e.get("duration_ms") or 0
        parse_ms += e.get("parse_ms") or 0
    tool_s = sum(by_cat.values()) / 1000
    llm_s = usage.get("llm_latency_s") or 0.0
    wall = meta.get("wall_time_s") or 0.0
    return {
        "wall_time_s": wall,
        "llm_time_s": round(llm_s, 4),
        "tool_time_s": round(tool_s, 4),
        "navigate_time_s": round(by_cat["navigate"] / 1000, 4),
        "search_time_s": round(by_cat["search"] / 1000, 4),
        "read_time_s": round(by_cat["read"] / 1000, 4),
        "parse_time_s": round(parse_ms / 1000, 4),
        "overhead_s": round(max(0.0, wall - llm_s - tool_s), 4),
    }


def _tokens(usage: dict, tool_output_chars: int) -> dict:
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    total = usage.get("total_tokens")
    if total is None and (inp is not None or out is not None):
        total = (inp or 0) + (out or 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "tool_output_tokens": tool_output_chars // 4,
        "cost_usd": usage.get("cost_usd"),
    }


def classify_failure(outcome: dict, io: dict, answer: dict, meta: dict, manifest: dict, workspace: Path) -> dict:
    io["recovered"] = io["trap_exposed"] and bool(outcome["success"])
    if outcome["success"]:
        return {"failure_type": None, "failure_detail": None}

    usage = meta.get("usage") or {}
    err = meta.get("agent_error") or ""
    task_type = manifest["task"]["type"]
    schema_keys = list(manifest["task"]["answer_schema"])
    fields = outcome.get("fields") or {}

    if usage.get("hit_step_limit") or "timeout" in str(err).lower():
        return {"failure_type": "timeout", "failure_detail": "step_limit" if usage.get("hit_step_limit") else "timeout"}
    if err:
        return {"failure_type": "agent_error", "failure_detail": str(err).strip().splitlines()[-1][:200]}
    if task_type == "workflow" and fields.get("file_exists") is False:
        return {"failure_type": "write_failure", "failure_detail": manifest["task"]["output_path"]}
    if task_type == "workflow" and fields.get("header") is False:
        return {"failure_type": "output_format_failure", "failure_detail": "csv_header"}
    if task_type != "workflow" and (not answer or any(k not in answer for k in schema_keys)):
        return {"failure_type": "output_format_failure", "failure_detail": "missing_schema_keys"}

    if io["required_recall_read"] < 1.0:
        detail = "path_hallucination" if io["hallucinated_paths"] else "required_unread"
        return {"failure_type": "retrieval_failure", "failure_detail": detail}

    match = _decoy_match(task_type, answer, io, outcome, manifest, workspace)
    if match:
        return match
    return {"failure_type": "reasoning_failure", "failure_detail": None}


def _value_eq(key: str, pred, decoy) -> bool:
    if key in ("outstanding_balance", "annual_value", "amount_due"):
        return _money_eq(pred, decoy)
    if key == "renewal_date":
        left = parse_date(pred) if isinstance(pred, str) else None
        if isinstance(decoy, str):
            right = parse_date(decoy) or (date.fromisoformat(decoy) if len(decoy) >= 8 else None)
        else:
            right = None
        return left is not None and left == right
    if key == "unpaid_invoices":
        pred_set = {_norm(x).upper() for x in pred} if isinstance(pred, list) else set()
        decoy_set = {_norm(x).upper() for x in decoy} if isinstance(decoy, list) else set()
        return bool(pred_set) and pred_set == decoy_set
    return pred is not None and decoy is not None and _norm(pred) == _norm(decoy)


def _workflow_pred_rows(workspace: Path, output_path: str) -> set[tuple[str, float]] | None:
    path = workspace / output_path
    if not path.is_file():
        return None
    reader = csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines())
    pred = set()
    for row in reader:
        row = {(k or "").strip().lower(): v for k, v in row.items()}
        amount = parse_money(row.get("amount_due") or "")
        if row.get("invoice_id") and amount is not None:
            pred.add((row["invoice_id"].strip().upper(), round(amount, 2)))
    return pred


def _decoy_matches(task_type: str, answer: dict, decoy: dict, outcome: dict, workspace: Path, output_path: str | None) -> bool:
    if task_type == "workflow":
        pred = _workflow_pred_rows(workspace, output_path or "")
        if pred is None:
            return False
        rows = decoy.get("rows") or []
        try:
            decoy_rows = {(r["invoice_id"].upper(), round(float(r["amount_due"]), 2)) for r in rows}
        except (KeyError, TypeError, ValueError):
            return False
        return bool(decoy_rows) and pred == decoy_rows
    wrong = {k for k, ok in (outcome.get("fields") or {}).items() if not ok}
    for k, v in decoy.items():
        if k == "invoice_id" and _value_eq("unpaid_invoices", answer.get("unpaid_invoices"), [v]):
            return True
        if k in answer and _value_eq(k, answer.get(k), v) and (k in wrong or k not in (outcome.get("fields") or {})):
            return True
    return False


def _decoy_match(task_type: str, answer: dict, io: dict, outcome: dict, manifest: dict, workspace: Path) -> dict | None:
    decoys = manifest.get("decoy_answers") or {}
    roles = {f["doc_id"]: f["role"] for f in manifest["files"]}
    output_path = manifest["task"].get("output_path")
    read_ids = io.get("read_doc_ids") or []
    for role, ftype in (("trap", "stale_version_confusion"), ("near", "distractor_confusion")):
        for doc_id in read_ids:
            if roles.get(doc_id) != role or doc_id not in decoys:
                continue
            if _decoy_matches(task_type, answer, decoys[doc_id], outcome, workspace, output_path):
                return {"failure_type": ftype, "failure_detail": doc_id}
    return None


def find_runs(root: str | Path) -> list[Path]:
    return sorted(Path(display_path(p.parent)) for p in long_path(root).rglob("meta.json")
                  if (p.parent / "answer.json").exists())


def find_envs(root: str | Path) -> list[Path]:
    lroot = long_path(root)
    if (lroot / "manifest.json").exists():
        return [Path(display_path(lroot))]
    return sorted(Path(display_path(p.parent)) for p in lroot.rglob("manifest.json")
                  if (p.parent / "workspace").is_dir())


def select_envs(env_dirs: list[Path], *, replicates: int | None = None, only: str | None = None) -> list[Path]:
    """Keep a seed-paired slice: ``sweep.replicate < N`` and optionally one swept variable."""
    if replicates is None and not only:
        return env_dirs
    out = []
    for env in env_dirs:
        m = json.loads((long_path(env) / "manifest.json").read_text(encoding="utf-8"))
        sweep = m.get("sweep") or {}
        if replicates is not None:
            r = sweep.get("replicate")
            if r is None or r >= replicates:
                continue
        if only:
            vars_ = sweep.get("vars") or {}
            if only not in vars_:
                continue
        out.append(env)
    return out


def flatten_row(metrics: dict, manifest: dict, meta: dict | None = None) -> dict:
    row = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    row["fields_correct"] = json.dumps(metrics["outcome"].get("fields", {}))
    for k, v in (metrics.get("usage") or {}).items():
        if not isinstance(v, list):
            row[f"usage.{k}"] = v
    for k, v in manifest["config"].items():
        row[f"cfg.{k}"] = "+".join(v) if isinstance(v, list) else v
    for k, v in manifest["stats"].items():
        if not isinstance(v, dict):
            row[f"stat.{k}"] = v
    sweep = manifest.get("sweep") or {}
    row["sweep.vars"] = ";".join(f"{k}={v}" for k, v in (sweep.get("vars") or {}).items())
    row["sweep.replicate"] = sweep.get("replicate")
    cond = manifest.get("condition") or {}
    row["condition"] = cond.get("label") or row["sweep.vars"] or "default"
    row["config_hash"] = cond.get("config_hash")
    for k, v in manifest["seeds"].items():
        row[f"seed.{k}"] = v
    if meta:
        row["timestamp"] = meta.get("timestamp")
        row["git_commit"] = meta.get("git_commit")
        row["fsbench_version"] = meta.get("fsbench_version")
        desc = meta.get("agent_describe") or {}
        row["agent_prompt_version"] = desc.get("agent_prompt_version")
        row["model"] = meta.get("model") or desc.get("model")
    return row


def evaluate_runs(root: str | Path, out_csv: str | Path | None = None) -> list[dict]:
    rows = []
    for run in find_runs(root):
        metrics = evaluate_run(run)
        (long_path(run) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        meta = json.loads((long_path(run) / "meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((long_path(meta["env_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        rows.append({"run_dir": str(run), **flatten_row(metrics, manifest, meta)})
    if out_csv and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def _sweep_vars(row: dict) -> dict[str, str]:
    raw = row.get("sweep.vars") or ""
    if not raw:
        return {}
    out = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def seed_sets(rows: list[dict]) -> dict[tuple, dict[str, set]]:
    """(agent, toolset, task_type, variable) -> {condition_value: set(world_seed)}."""
    groups: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        seed = r.get("seed.world")
        for var, val in _sweep_vars(r).items():
            groups[(r.get("agent"), r.get("toolset"), r.get("task_type"), var)][val].add(seed)
    return groups


def balance_warnings(rows: list[dict]) -> list[str]:
    warnings = []
    for key, by_val in seed_sets(rows).items():
        sets = list(by_val.values())
        if len(sets) < 2:
            continue
        union, inter = set.union(*sets), set.intersection(*sets)
        if union != inter:
            missing = {val: sorted(union - seeds, key=lambda x: (x is None, x))
                       for val, seeds in by_val.items() if union - seeds}
            warnings.append(
                f"unbalanced seeds for {key[2]} {key[3]} [{key[0]}/{key[1]}]: "
                f"missing {missing}"
            )
    return warnings


def mixed_version_warnings(rows: list[dict]) -> list[str]:
    warnings = []
    for field in ("fsbench_version", "agent_prompt_version", "model"):
        vals = sorted({r.get(field) for r in rows if r.get(field)})
        if len(vals) > 1:
            warnings.append(f"summary mixes {field} values: {vals}")
    return warnings


def keep_paired(rows: list[dict]) -> list[dict]:
    """Drop seeds that are not present in every condition of each swept variable."""
    sets = seed_sets(rows)
    intersections = {key: set.intersection(*by_val.values()) if by_val else set()
                     for key, by_val in sets.items()}
    keep = []
    for r in rows:
        vars_ = _sweep_vars(r)
        if not vars_:
            keep.append(r)
            continue
        if all(r.get("seed.world") in intersections[(r.get("agent"), r.get("toolset"), r.get("task_type"), var)]
               for var in vars_):
            keep.append(r)
    return keep


def summarize(rows: list[dict], by: tuple[str, ...] = ("agent", "toolset", "task_type", "condition")) -> list[dict]:
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
        counts = Counter(r.get("failure_type") for r in rs if r.get("failure_type"))
        for ft in FAILURE_TYPES:
            s[f"fail.{ft}"] = counts.get(ft, 0)
        out.append(s)
    return out


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return 0.0, 0.0
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var / n)
