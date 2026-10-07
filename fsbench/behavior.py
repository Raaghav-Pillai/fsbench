"""Deterministic trace annotations; labels describe observations, not hidden thoughts."""
import json
import posixpath
from collections import Counter, defaultdict
from pathlib import Path

from fsbench.paths import long_path
from fsbench.benchmark import load_manifest

BEHAVIOR_METRICS = ("searches_before_evidence", "distractors_before_evidence", "stale_reads_before_evidence",
                    "duplicate_read_count", "backtrack_count", "answered_without_evidence", "recovered_tool_error")


def trajectory(events, manifest, *, answered=True):
    files = {f["path"]: f for f in manifest["files"]}
    required = set(manifest["required_doc_ids"])
    found, read_ids, visits = set(), set(), {}
    counters = Counter()
    annotated = []
    last_directory = None
    for event in events:
        labels, path_labels = [], {}
        tool = event["tool"]
        before = not found
        if tool in ("search_files", "grep", "search_index", "find", "glob"):
            labels.append("INDEX_SEARCH" if tool == "search_index" else "SEARCH")
            if before:
                counters["searches_before_evidence"] += 1
        elif tool == "stat_file":
            labels.append("METADATA_LOOKUP")
        elif event.get("category") == "navigate":
            labels.append("DIRECTORY_NAVIGATION")
            if event.get("ok"):
                raw = event.get("args", {}).get("path", event.get("cwd", "/workspace"))
                directory = posixpath.normpath(raw if raw.startswith("/") else
                    posixpath.join(event.get("cwd", "/workspace"), raw))
                if tool == "cd":
                    directory = event.get("cwd", directory)
                if directory != last_directory and visits.get(directory) == len(found):
                    counters["backtrack_count"] += 1
                visits[directory] = len(found)
                last_directory = directory
        for path in event.get("read", []):
            f = files.get(path)
            if not f:
                continue
            tags = []
            doc_id = f["doc_id"]
            if doc_id in required:
                tags.append("REQUIRED_FILE_READ")
            else:
                tags.append("DISTRACTOR_READ")
                if before:
                    counters["distractors_before_evidence"] += 1
            if f["role"] == "trap":
                tags.append("STALE_FILE_READ")
                if before:
                    counters["stale_reads_before_evidence"] += 1
            if f["role"] == "near":
                tags.append("NEAR_MISS_READ")
            if f.get("copy", 0) or doc_id in read_ids:
                tags.append("DUPLICATE_READ")
            if doc_id in read_ids:
                counters["duplicate_read_count"] += 1
            read_ids.add(doc_id)
            found.update({doc_id} & required)
            path_labels[path] = tags
            labels.extend(tags)
        if event.get("missing"):
            labels.append("INVALID_PATH")
        if event.get("written"):
            labels.append("WRITE")
        annotated.append({**event, "labels": list(dict.fromkeys(labels)), "path_labels": path_labels})
    if answered:
        annotated.append({"step": len(events)+1, "tool": "answer", "labels": ["FINAL_ANSWER"]})
    metrics = {k: counters[k] for k in BEHAVIOR_METRICS if k not in ("answered_without_evidence", "recovered_tool_error")}
    metrics["answered_without_evidence"] = answered and not required.issubset(found)
    # Tool-error recovery is observable continuation to a successful tool action.
    error_steps = [i for i, e in enumerate(events) if not e.get("ok", True)]
    recovered = [i for i in error_steps if any(e.get("ok") for e in events[i+1:])]
    metrics.update(tool_error_count=len(error_steps), recovered_tool_error_count=len(recovered),
                   recovered_tool_error=bool(recovered))
    return annotated, metrics


def failure_decomposition(legacy, outcome, io, answer, meta, manifest, workspace, events):
    """Keep the legacy failure_type API; expose finer task_failure_type and family."""
    from fsbench.evaluate import _decoy_match
    if legacy["failure_type"] == "required_tool_not_used":
        kind = "required_tool_not_used"
    elif outcome["success"]:
        kind = None
    elif (meta.get("usage") or {}).get("provider_errors") or meta.get("agent_error"):
        kind = "provider_failure" if (meta.get("usage") or {}).get("provider_errors") else "agent_error"
    elif (meta.get("usage") or {}).get("hit_step_limit"):
        kind = "step_limit"
    elif legacy["failure_type"] in ("output_format_failure", "write_failure"):
        kind = legacy["failure_type"]
    else:
        match = _decoy_match(manifest["task"]["type"], answer, io, outcome, manifest, workspace)
        if match:
            kind = "near_miss_confusion" if match["failure_type"] == "distractor_confusion" else match["failure_type"]
        elif io["required_recall_read"] < 1:
            if io["hallucinated_paths"]:
                kind = "path_hallucination"
            elif events and not events[-1].get("ok", True):
                kind = "tool_misuse"
            else:
                kind = "retrieval_failure"
        else:
            kind = "reasoning_failure"
    family = "source_selection_failure" if kind in ("near_miss_confusion", "stale_version_confusion") else kind
    return {"task_failure_type": kind, "failure_family": family}


def inspect_run(run):
    from fsbench.evaluate import evaluate_run, load_trace
    run = long_path(run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    m = load_manifest(meta["env_dir"])
    metrics = evaluate_run(run)
    events, _ = trajectory(load_trace(run / "trace.jsonl"), m, answered=not meta.get("agent_error"))
    lines = ["TASK", m["task"]["prompt"], "", "OUTCOME",
             "Correct" if metrics["success"] else str(metrics["task_failure_type"]), "", "TRAJECTORY"]
    for e in events:
        lines.append(f"{e['step']}. {e['tool']}({json.dumps(e.get('args', {}), ensure_ascii=False)})")
        lines.append("   " + ", ".join(e["labels"]))
        for path, labels in e.get("path_labels", {}).items():
            lines.append(f"   {path}: {', '.join(labels)}")
        if e.get("error"):
            lines.append("   Error: " + e["error"])
    lines.extend(["", "SUMMARY"])
    for field in ("n_calls", "files_read", "steps_to_first_required_evidence", *BEHAVIOR_METRICS):
        lines.append(f"{field}: {metrics[field]}")
    return "\n".join(lines)


def analyze_behavior(rows, group_by):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, r.get("cfg." + k)) for k in group_by)
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda item: str(item[0])):
        record = {**dict(zip(group_by, key)), "n_runs": len(rs)}
        for metric in BEHAVIOR_METRICS:
            vals = [float(r[metric]) for r in rs if r.get(metric) is not None]
            record["mean_" + metric] = sum(vals)/len(vals) if vals else None
        for kind, n in Counter(r.get("task_failure_type") or "success" for r in rs).items():
            record["failure." + kind] = n
        out.append(record)
    return out
