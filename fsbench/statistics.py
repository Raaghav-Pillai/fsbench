"""World-paired analysis. Repeated trials are averaged before resampling worlds."""
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, variance

from fsbench.interaction import bootstrap_difference

DIMENSIONS = ("condition_id", "model", "toolset", "tool_policy", "trial_index")
METRICS = ("success", "n_calls", "files_read", "total_tokens", "cost_usd", "wall_time_s")


def world(r):
    return tuple(r.get("seed." + k, r.get(k + "_seed")) for k in ("world", "task", "layout"))


def design_pairing(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("design_id"), r.get("task_type"))].append(r)
    kept, warnings = [], []
    for group, records in groups.items():
        expected_specs = []
        for r in records:
            cells = r["expected_cells"]
            cells = json.loads(cells) if isinstance(cells, str) else cells
            expected_specs.extend(c for c in cells if c["task"] == group[1])
        worlds = defaultdict(list)
        for r in records:
            worlds[world(r)].append(r)
        for seed, trials in worlds.items():
            expected = {tuple(c.get(k) for k in DIMENSIONS) for c in expected_specs
                        if "world_seed" not in c or tuple(c[k] for k in ("world_seed","task_seed","layout_seed"))==seed}
            counts = Counter(tuple(r.get(k) for k in DIMENSIONS) for r in trials)
            missing, extra = expected - counts.keys(), counts.keys() - expected
            duplicates = [c for c, n in counts.items() if n > 1]
            mismatches = []
            for condition in {c[0] for c in expected}:
                same = [r for r in trials if r["condition_id"] == condition]
                for field in ("input_sha256", "workspace_sha256"):
                    hashes = {r.get(field) for r in same}
                    if len(hashes) != 1 or None in hashes:
                        mismatches.append((condition, field))
            if missing or extra or duplicates or mismatches:
                warnings.append(f"incomplete design {group}, seeds={seed}: missing cells={sorted(missing)}, "
                                f"unexpected cells={sorted(extra)}, duplicates={duplicates}, input mismatches={mismatches}")
            else:
                kept.extend(trials)
    return kept, warnings


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p, d = successes / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, center-half), min(1, center+half)


def analyze_statistics(rows, *, baseline="clean", bootstrap_seed=7):
    from fsbench.interaction import factorial_pairing
    rows, warnings = factorial_pairing(rows)
    # Separate experiments/tasks/settings. The cell keys retain models and policies.
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("design_id") or r.get("experiment"), r["task_type"],
                r.get("filename_noise_version"))].append(r)
    summaries, contrasts, variations, factorial = [], [], [], []
    for group, records in groups.items():
        cells = defaultdict(list)
        labels = defaultdict(set)
        for r in records:
            if r.get("condition_id"):
                labels[r["condition"]].add(r["condition_id"])
        if any(len(ids)>1 for ids in labels.values()):
            raise ValueError("different configurations share a condition label; generate unique --label values")
        for r in records:
            cells[(r.get("model") or r["agent"], r["toolset"], r.get("tool_policy", "naturalistic"), r["condition"])].append(r)
        prefix = {"experiment": group[0], "task": group[1]}
        for metric in METRICS:
            means = {}
            for cell, trials in cells.items():
                by_world = defaultdict(list)
                for r in trials:
                    by_world[world(r)].append(r.get(metric))
                # A partially reported metric is not treated as a complete trial average.
                good = {s: list(map(float, vs)) for s, vs in by_world.items() if all(v is not None for v in vs)}
                means[cell] = {s: mean(vs) for s, vs in good.items()}
                vals = list(means[cell].values())
                boot = bootstrap_difference(vals, seed=bootstrap_seed)
                rec = {**prefix, "model": cell[0], "toolset": cell[1], "tool_policy": cell[2],
                       "condition": cell[3], "metric": metric, "n_worlds": len(vals),
                       "n_trials": sum(map(len, good.values())), "mean": boot["mean_difference"],
                       "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                       "ci_method": "world bootstrap, 10000 resamples",
                       "within_seed_variance": mean([variance(v) for v in good.values() if len(v)>1])
                            if any(len(v)>1 for v in good.values()) else None,
                       "between_seed_variance": variance(vals) if len(vals)>1 else None}
                if metric == "success" and all(len(v)==1 for v in good.values()):
                    rec["ci_low"], rec["ci_high"] = wilson(sum(vals), len(vals))
                    rec["ci_method"] = "Wilson, independent worlds"
                rec["boundary_interval"] = metric == "success" and bool(vals) and len(set(vals)) == 1 and vals[0] in (0, 1)
                summaries.append(rec)
            for a, b in combinations(sorted(cells), 2):
                kind = None
                if a[:3] == b[:3] and baseline in (a[3], b[3]):
                    a, b = (a, b) if a[3] == baseline else (b, a)
                    kind = "robustness_delta"
                elif a[3] == b[3] and (a[0] == b[0] or a[1:3] == b[1:3]):
                    kind = "interface_comparison" if a[0] == b[0] else "model_comparison"
                if kind:
                    common = sorted(means[a].keys() & means[b].keys())
                    diffs = [means[b][s] - means[a][s] for s in common]
                    sd = math.sqrt(variance(diffs)) if len(diffs)>1 else 0
                    contrasts.append({**prefix, "kind": kind, "metric": metric,
                        "from_cell": json.dumps(a), "to_cell": json.dumps(b),
                        **bootstrap_difference(diffs, seed=bootstrap_seed),
                        "paired_standardized_difference": mean(diffs)/sd if sd else None})
            arms = {c[:3] for c in cells}
            for arm in sorted(arms):
                mus = [mean(v.values()) for c, v in means.items() if c[:3] == arm and v]
                variations.append({**prefix, "model": arm[0], "toolset": arm[1], "tool_policy": arm[2],
                    "metric": metric, "between_condition_variance": variance(mus) if len(mus)>1 else None})
            # Two-factor endpoint interaction, holding every other config field fixed.
            factors = sorted({part.split("=", 1)[0] for r in records
                              for part in (r.get("sweep.vars") or "").split(";") if "=" in part})
            for x, y in combinations(factors, 2):
                config_key = lambda k: "cfg.mime_types" if k == "mime_diversity" else "cfg." + k
                fixed_groups = defaultdict(dict)
                for cell, trials in cells.items():
                    r = trials[0]
                    fixed = tuple(sorted((k, str(v)) for k, v in r.items() if k.startswith("cfg.")
                                         and k not in (config_key(x), config_key(y))))
                    try:
                        xy = (float(r[config_key(x)]), float(r[config_key(y)]))
                    except (TypeError, ValueError, KeyError):
                        continue
                    fixed_groups[(cell[:3], fixed)][xy] = means[cell]
                for (arm, fixed), grid in fixed_groups.items():
                    xs, ys = sorted({p[0] for p in grid}), sorted({p[1] for p in grid})
                    if len(xs)<2 or len(ys)<2:
                        continue
                    corners = [(xs[0],ys[0]), (xs[-1],ys[0]), (xs[0],ys[-1]), (xs[-1],ys[-1])]
                    if any(c not in grid for c in corners):
                        continue
                    common = set.intersection(*(set(grid[c]) for c in corners))
                    diffs = [grid[corners[3]][s] - grid[corners[2]][s] - grid[corners[1]][s] + grid[corners[0]][s]
                             for s in sorted(common)]
                    factorial.append({**prefix, "model": arm[0], "toolset": arm[1], "tool_policy": arm[2],
                        "factor_x": x, "factor_y": y, "corners": json.dumps(corners), "fixed_config": json.dumps(fixed),
                        "metric": metric, **bootstrap_difference(diffs, seed=bootstrap_seed)})
    return {"warnings": warnings, "cells": summaries, "contrasts": contrasts,
            "variation": variations, "factorial": factorial}


def write_statistics(rows, out, *, baseline="clean", bootstrap_seed=7):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = analyze_statistics(rows, baseline=baseline, bootstrap_seed=bootstrap_seed)
    if not any(r.get("condition") == baseline for r in rows):
        tables["warnings"].append(f"baseline condition {baseline!r} absent; no robustness deltas computed")
    payload = {"baseline": baseline, "bootstrap_seed": bootstrap_seed, "resamples": 10000,
        "method": "Average complete trials within each world/cell, then resample paired worlds. "
                  "Repeated-trial success uses world bootstrap; boundary intervals can be degenerate and do not prove certainty. "
                  "Variance components are descriptive, not a fitted random-effects model.", **tables}
    path = out / "statistics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in ("cells", "contrasts", "variation", "factorial"):
        write_csv(out / ("statistics_" + name + ".csv"), tables[name])
    return path
