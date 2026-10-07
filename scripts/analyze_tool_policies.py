"""Audit and report the completed 240-run index-first experiment, without model calls."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsbench.behavior import analyze_behavior
from fsbench.evaluate import evaluate_runs, find_runs, load_trace
from fsbench.experiment import validate_environment
from fsbench.interaction import compare_trace, write_interaction
from fsbench.paths import long_path
from fsbench.plot import plot_runs
from fsbench.statistics import write_csv, write_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/exp_noise_policies_v1")
    args = ap.parse_args()
    root = Path(args.runs)
    plan = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    schedule = {c["execution_order"]: c for c in plan["schedule"]}
    runs = find_runs(root)
    if len(runs) != len(schedule):
        raise ValueError(f"experiment incomplete: {len(runs)}/{len(schedule)} runs")
    source_hashes, seen = {}, set()
    index_uses, tool_errors = Counter(), Counter()
    usage_by_arm = defaultdict(Counter)
    metas = []
    for run in runs:
        path = long_path(run)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        order = meta["execution_order"]
        if order in seen or order not in schedule:
            raise ValueError("duplicate/unplanned execution order")
        seen.add(order)
        cell = schedule[order]
        for field in ("input_sha256", "workspace_sha256", "env_id", "toolset", "tool_policy", "world_seed", "filename_noise"):
            if meta[field] != cell[field]:
                raise ValueError(f"run differs from plan: {run}, {field}")
        env = meta["env_dir"]
        if env not in source_hashes:
            source_hashes[env] = validate_environment(env)
        if source_hashes[env] != {k: meta[k] for k in ("input_sha256", "workspace_sha256")}:
            raise ValueError(f"source environment changed: {env}")
        events = load_trace(path / "trace.jsonl")
        arm = meta["toolset"] + "/" + meta["tool_policy"]
        successful_index = [e for e in events if e["tool"] == "search_index" and e["ok"]]
        index_uses[arm] += bool(successful_index)
        tool_errors[arm] += sum(not e["ok"] for e in events)
        usage_by_arm[arm].update(e["tool"] for e in events)
        if meta["tool_policy"] == "index_first":
            if not successful_index or any(e["read"] for e in events if e["step"] < successful_index[0]["step"]):
                raise ValueError(f"index-first policy violated: {run}")
            if events[0]["tool"] != "search_index":
                raise ValueError(f"index-first initial action mismatch: {run}")
        metas.append(meta)
    rows = evaluate_runs(root, root / "results.csv", paired=True)
    if len(rows) != len(schedule):
        raise ValueError("some scheduled trials failed pairing")
    write_interaction(rows, root)
    write_statistics(rows, root, baseline="filename_noise=0")
    write_csv(root / "behavior.csv", analyze_behavior(rows, ["model", "filename_noise", "toolset", "tool_policy"]))
    plot_runs(rows, "filename_noise", root / "plots")
    (root / "trace_seed7_noise09.txt").write_text(compare_trace(root, 7, .9), encoding="utf-8")
    audit = {"runs": len(rows), "successes": sum(r["success"] for r in rows),
        "valid": sum(r["valid"] for r in rows), "environments_unchanged": len(source_hashes),
        "paired_worlds": len({(r["seed.world"], r["seed.task"], r["seed.layout"]) for r in rows}),
        "index_used_runs": dict(index_uses), "tool_errors": dict(tool_errors),
        "tool_usage": {k: dict(v) for k, v in usage_by_arm.items()},
        "reported_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 8),
        "first_completed_at": min(m["timestamp"] for m in metas),
        "last_completed_at": max(m["timestamp"] for m in metas),
        "execution_order_seed": plan["experiment_order_seed"],
        "notes": "Single trial per cell. Runner was started after policy/index validation, before subsequent v0.2 refactors. "
                 "Its loaded policy and agent code remained fixed. Development and tests overlapped collection, so wall time is exploratory."}
    (root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
