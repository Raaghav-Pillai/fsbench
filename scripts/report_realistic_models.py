"""Report a completed paired realistic study, clustering sibling tasks by world."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import mean, median

from fsbench.experiment import validate_environment
from fsbench.benchmark import load_manifest
from fsbench.realistic import semantic_equal
from fsbench.statistics import write_csv

METRICS = ("success", "evidence_precision", "evidence_recall", "n_calls", "files_read",
           "total_tokens", "cost_usd", "wall_time_s")


def bootstrap_worlds(values, domains, seed=7, resamples=10000):
    """Keep domain balance fixed; resample worlds, retaining sibling-task dependence."""
    buckets = defaultdict(list)
    for world, value in sorted(values.items()):
        buckets[domains[world]].append(float(value))
    rng = random.Random(seed)
    n = len(values)
    if not n:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_worlds": 0}
    samples = sorted(sum(sum(rng.choices(v, k=len(v))) for v in buckets.values()) / n
                     for _ in range(resamples))
    return {"mean": mean(values.values()), "ci_low": samples[int(.025 * resamples)],
            "ci_high": samples[int(.975 * resamples)], "n_worlds": n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root, out = Path(a.runs), Path(a.out)
    plan = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    expected = {(c["env_id"], c["model"], c["trial_index"]): c for c in plan["schedule"]}
    rows, seen, audit = [], set(), []
    for path in sorted(root.rglob("metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
        key = (m["env_id"], m["model"], meta["trial_index"])
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected run: {key}")
        seen.add(key)
        cell = expected[key]
        for field in ("input_sha256", "workspace_sha256", "execution_order"):
            if meta[field] != cell[field]:
                raise ValueError(f"run integrity mismatch: {key}, {field}")
        if meta["design_id"] != plan["design_id"]:
            raise ValueError(f"different experimental design: {key}")
        rows.append({**m, "study_family": cell["study_family"], "trial_index": meta["trial_index"],
                     "run_path": str(path.parent.resolve())})
        if not m["success"]:
            manifest = load_manifest(meta["env_dir"])
            answer = json.loads((path.parent / "answer.json").read_text(encoding="utf-8"))
            predicted = answer.get("answer") if isinstance(answer, dict) else answer
            expected_answer = manifest["ground_truth"].get("answer")
            def leaves(value):
                if isinstance(value, dict):
                    return [leaf for v in value.values() for leaf in leaves(v)]
                if isinstance(value, list):
                    return [leaf for v in value for leaf in leaves(v)]
                return [value]
            wrapper_match = (not isinstance(expected_answer, (dict, list))
                             and isinstance(predicted, (dict, list))
                             and any(semantic_equal(v, expected_answer) for v in leaves(predicted)))
            audit.append({"model": m["model"], "env_id": m["env_id"], "task_template": m["task_template"],
                          "failure_type": m.get("task_failure_type") or m.get("failure_type"),
                          "potential_scalar_wrapper_match": wrapper_match,
                          "prompt": manifest["task"]["prompt"], "expected_answer": expected_answer,
                          "predicted_answer": predicted, "raw_answer": meta.get("raw_answer"),
                          "required_recall_read": m.get("required_recall_read"),
                          "evidence_recall": m.get("evidence_recall"), "run_path": str(path.parent.resolve())})
    if seen != expected.keys():
        raise ValueError(f"incomplete study: {len(seen)}/{len(expected)} cells")
    env_cells = {c["env_id"]: c for c in plan["schedule"]}
    for c in env_cells.values():
        if validate_environment(c["env_dir"]) != {k: c[k] for k in ("workspace_sha256", "input_sha256")}:
            raise ValueError(f"source environment changed: {c['env_id']}")
    out.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    domains = {r["world_id"]: r["workspace_domain"] for r in rows}
    groups = {m: [r for r in rows if r["model"] == m] for m in models}
    world_values = {}
    intervals, summaries = [], []
    for model, rs in groups.items():
        summary = {"model": model, "runs": len(rs), "successes": sum(r["success"] for r in rs),
                   "provider_failures": sum(bool(r.get("agent_error")) for r in rs),
                   "potential_scalar_wrapper_false_negatives": sum(r["potential_scalar_wrapper_match"] for r in audit if r["model"] == model)}
        for metric in METRICS:
            worlds = defaultdict(list)
            for r in rs:
                worlds[r["world_id"]].append(r.get(metric))
            vals = {w: mean(v) for w, v in worlds.items() if all(x is not None for x in v)}
            world_values[model, metric] = vals
            interval = bootstrap_worlds(vals, domains)
            intervals.append({"model": model, "metric": metric, **interval})
            valid = [r[metric] for r in rs if r.get(metric) is not None]
            summary["mean_" + metric] = mean(valid) if valid else None
            summary["median_" + metric] = median(valid) if valid else None
            summary["reported_" + metric] = len(valid)
        summary["total_cost_usd"] = sum(r["cost_usd"] for r in rs if r.get("cost_usd") is not None)
        summaries.append(summary)
    contrasts = []
    for left, right in combinations(models, 2):
        for metric in METRICS:
            lv, rv = world_values[left, metric], world_values[right, metric]
            delta = {w: rv[w] - lv[w] for w in lv.keys() & rv.keys()}
            result = bootstrap_worlds(delta, domains)
            contrasts.append({"from_model": left, "to_model": right, "metric": metric, **result,
                              "interval_includes_zero": (result["ci_low"] <= 0 <= result["ci_high"])
                              if result["ci_low"] is not None else None})
    breakdown = []
    for dimension in ("workspace_domain", "study_family", "difficulty_level"):
        for model in models:
            for value in sorted({r[dimension] for r in groups[model]}):
                rs = [r for r in groups[model] if r[dimension] == value]
                rec = {"model": model, "dimension": dimension, "value": value, "n": len(rs)}
                for metric in METRICS:
                    vals = [r[metric] for r in rs if r.get(metric) is not None]
                    rec[metric] = mean(vals) if vals else None
                breakdown.append(rec)
    failure_counts = Counter((r["model"], r.get("task_failure_type") or r.get("failure_type") or "success") for r in rows)
    failures = [{"model": m, "failure_type": f, "count": n} for (m, f), n in sorted(failure_counts.items())]
    payload = {"design_id": plan["design_id"], "runs": len(rows), "worlds": len(domains),
               "tasks": len(env_cells), "models": models, "source_environments_unchanged": True,
               "method": "10000 bootstrap resamples, seed 7; average sibling tasks within world, then resample worlds within each domain. Paired differences are to_model minus from_model.",
               "limitations": ["Raw correctness has known schema-related false negatives: prompts permit descriptive objects but the evaluator sometimes expects a scalar. Do not treat raw scores as a reliable model ranking.",
                               "Potential wrapper matches are diagnostic flags, not automatically rescored successes; a matching value could have the wrong semantic role.",
                               "One trial per task/model; no model stochasticity estimate.",
                               "Authored difficulty groups contain different worlds; differences are observational, not causal robustness deltas.",
                               "Boundary bootstrap intervals may be degenerate and do not imply certainty.",
                               "Tables average available run-level observations; interval estimates and plots exclude a world if any sibling task lacks that metric. Effective world counts are recorded in intervals.csv.",
                               "No human baselines or established external validity.",
                               "Latency includes provider variability; automatic routing is not a pinned provider.",
                               "First seven runs were serial; remaining runs used five workers. One interrupted attempt was archived and restarted, with unknown partial cost. Planned order is fixed; actual timing varies."],
               "summaries": summaries, "intervals": intervals, "contrasts": contrasts,
               "breakdowns": breakdown, "failures": failures}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "failure_audit_private.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for name, data in (("summary", summaries), ("intervals", intervals), ("contrasts", contrasts),
                       ("breakdowns", breakdown), ("failures", failures)):
        write_csv(out / f"{name}.csv", data)
    lines = ["# Realistic FSBench: five-model validation study", "",
             f"Completed {len(rows)} runs: {len(env_cells)} tasks across {len(domains)} worlds, on {len(models)} models.",
             "All models used identical source workspaces, files/naturalistic tools, temperature 0, and a 40-turn limit. Order seed: 7. Source environments passed post-run integrity checks.", "",
             "**Scoring caveat:** success below uses the frozen evaluator and includes known schema-related false negatives. See the private failure audit; these results do not support a definitive model ranking.", "",
             "| Model | Raw correct | Evidence recall | Mean calls | Mean files read | Mean tokens | Mean seconds | Total USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        cost = f"${s['total_cost_usd']:.4f}" + (" (partial)" if s["reported_cost_usd"] < s["runs"] else "")
        lines.append(f"| {s['model']} | {s['successes']}/{s['runs']} | {s['mean_evidence_recall']:.1%} | {s['mean_n_calls']:.1f} | {s['mean_files_read']:.1f} | {s['mean_total_tokens']:.0f} | {s['mean_wall_time_s']:.1f} | {cost} |")
    reported_cost = sum(s["total_cost_usd"] for s in summaries)
    missing_cost = sum(s["runs"] - s["reported_cost_usd"] for s in summaries)
    lines += ["", f"Reported API cost: ${reported_cost:.6f}. Cost is missing for {missing_cost} completed runs and the interrupted attempt; this is not a complete billing total.",
              "", "Correctness and evidence attribution are separate. Lower cost or fewer calls can reflect premature answers, so efficiency alone does not establish better performance.",
              "", "## Scoring audit", "", "| Model | Potential scalar-wrapper false negatives |", "|---|---:|"]
    lines.extend(f"| {s['model']} | {s['potential_scalar_wrapper_false_negatives']} |" for s in summaries)
    lines += ["", "These flags identify matching scalar values inside a rejected object/list. They require semantic review and are not added to the raw success totals. The private audit contains evaluator answers and must remain private.",
              "", "![Model comparison](comparison.png)", "", "## Uncertainty and scope", "", payload["method"], "",
              "See contrasts.csv for paired differences and whether each 95% interval includes zero. Intervals are descriptive; no automatic significance claims are made.", ""]
    lines.extend("- " + limitation for limitation in payload["limitations"])
    lines += ["", "## Artifacts and reproduction", "",
              "- results.csv: complete paired run-level export, one directory above this report.",
              "- summary.csv, intervals.csv, contrasts.csv: descriptive metrics and paired uncertainty.",
              "- breakdowns.csv and behavior.csv: model/domain/family behavior.",
              "- failure_audit_private.json: private diagnostic records containing evaluator answers.",
              "- ../models/experiment.json and ../execution_events.jsonl: immutable design and execution events.",
              "", "```powershell",
              f"python -m fsbench evaluate --runs {a.runs} --paired --group-by model --out {out.parent / 'results.csv'}",
              f"python -m scripts.report_realistic_models --runs {a.runs} --out {a.out}",
              "```"]
    lines += ["", "## Failure counts", "", "| Model | Failure type | Count |", "|---|---|---:|"]
    lines.extend(f"| {r['model']} | {r['failure_type']} | {r['count']} |" for r in failures)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [m.split("/")[-1] for m in models]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plotted = (("success", "Raw task success (schema-sensitive)", 100), ("evidence_recall", "Evidence recall", 100),
               ("n_calls", "Tool calls", 1), ("files_read", "Files read", 1),
               ("total_tokens", "Total tokens", 1), ("wall_time_s", "Wall time (s)", 1))
    for ax, (metric, title, scale) in zip(axes.flat, plotted):
        data = [next(i for i in intervals if i["model"] == m and i["metric"] == metric) for m in models]
        vals = [r["mean"] * scale for r in data]
        errors = [[max(0, (r["mean"] - r["ci_low"]) * scale) for r in data],
                  [max(0, (r["ci_high"] - r["mean"]) * scale) for r in data]]
        ax.bar(range(len(models)), vals, yerr=errors, capsize=3, color=["#5b6ea6", "#478e95", "#c28c40", "#8968a8", "#ba646f"])
        ax.set_xticks(range(len(models)), labels, rotation=32, ha="right", fontsize=8)
        ax.set_title(title + (" (%)" if scale == 100 else ""))
        if any(r["n_worlds"] != len(domains) for r in data):
            ax.text(.02, .98, "Complete worlds: " + ", ".join(str(r["n_worlds"]) for r in data),
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if scale == 100:
            ax.set_ylim(0, 105)
    fig.suptitle("FSBench realistic-v1 candidate | 20 worlds, 60 tasks per model\n95% domain-stratified world bootstrap intervals", fontsize=14)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
