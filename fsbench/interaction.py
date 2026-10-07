"""Complete factorial pairing, paired bootstrap contrasts, and annotated traces."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from fsbench.paths import long_path
from fsbench.benchmark import load_manifest

CONTRAST_METRICS = ("n_calls", "files_read", "total_tokens", "cost_usd", "wall_time_s", "success")


def _group(r):
    fixed = tuple(sorted((k, str(v)) for k, v in r.items()
                         if k.startswith("cfg.") and k != "cfg.filename_noise"))
    return (r.get("experiment"), r.get("agent"), r.get("model"), r.get("task_type"),
            r.get("agent_settings"), r.get("filename_noise_version"), r.get("allow_writes"), fixed)


def _seed(r):
    return tuple(r.get(f"seed.{k}") for k in ("world", "task", "layout"))


def _cell(r):
    policy = r.get("tool_policy", "naturalistic")
    arm = r["toolset"] + ("/" + policy if policy != "naturalistic" else "")
    return (float(r["filename_noise"]), arm)


def factorial_pairing(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop whole seeds with missing/duplicate cells or different toolset inputs."""
    groups = defaultdict(list)
    passthrough = []
    from fsbench.statistics import design_pairing
    modern = [r for r in rows if r.get("design_version") == 2]
    modern_kept, modern_warnings = design_pairing(modern)
    for r in rows:
        if r.get("design_version") == 2:
            continue
        if r.get("expected_cells"):
            groups[_group(r)].append(r)
        else:
            passthrough.append(r)
    kept, warnings = list(passthrough) + modern_kept, list(modern_warnings)
    for group, rs in groups.items():
        expected = set()
        for r in rs:
            cells = json.loads(r["expected_cells"]) if isinstance(r["expected_cells"], str) else r["expected_cells"]
            expected.update(_cell(c) for c in cells)
        seeds = defaultdict(list)
        for r in rs:
            seeds[_seed(r)].append(r)
        for seed, trials in sorted(seeds.items(), key=lambda item: str(item[0])):
            counts = Counter(_cell(r) for r in trials)
            missing = expected - counts.keys()
            duplicate = [c for c, n in counts.items() if n != 1]
            input_mismatches = []
            for noise in {c[0] for c in expected}:
                same_env = [r for r in trials if float(r["filename_noise"]) == noise]
                for field in ("input_sha256", "workspace_sha256"):
                    hashes = {r.get(field) for r in same_env}
                    if len(hashes) != 1 or None in hashes:
                        input_mismatches.append((noise, field))
            if missing or duplicate or input_mismatches:
                warnings.append(f"incomplete factorial group {group[:4]}, seeds={seed}: "
                                f"missing cells={sorted(missing)}, duplicate cells={duplicate}, "
                                f"input mismatches={input_mismatches}")
            else:
                kept.extend(trials)
    return kept, warnings


def bootstrap_difference(differences: list[float], *, seed: int = 7, resamples: int = 10000) -> dict:
    if not differences:
        return {"n": 0, "mean_difference": None, "ci_low": None, "ci_high": None, "includes_zero": None}
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
    lo, hi = means[max(0, int(.025 * resamples) - 1)], means[max(0, int(.975 * resamples) - 1)]
    return {"n": n, "mean_difference": sum(differences) / n,
            "ci_low": lo, "ci_high": hi, "includes_zero": lo <= 0 <= hi}


def interaction_tables(rows: list[dict], *, bootstrap_seed: int = 7) -> dict[str, list[dict]]:
    from fsbench.evaluate import summarize

    rows, warnings = factorial_pairing(rows)
    if warnings:
        raise ValueError("interaction analysis needs complete verified cells: " + "; ".join(warnings))
    if not rows:
        raise ValueError("no complete paired trials")
    if len({_group(r) for r in rows}) != 1:
        raise ValueError("select one experiment/model/task/fixed configuration for interaction analysis")
    failure_rows = rows
    # Repeated calls are nested within a world; aggregate before paired contrasts.
    repeated = defaultdict(list)
    for r in rows:
        repeated[(_cell(r), _seed(r))].append(r)
    if any(len(rs)>1 for rs in repeated.values()):
        from fsbench.evaluate import SUMMARY_METRICS
        averaged = []
        for rs in repeated.values():
            if len({r.get("trial_index") for r in rs}) != len(rs):
                raise ValueError("duplicate seed/cell/trial in interaction analysis")
            row = dict(rs[0])
            for metric in set(CONTRAST_METRICS) | set(SUMMARY_METRICS):
                vs = [r.get(metric) for r in rs]
                row[metric] = sum(vs)/len(vs) if all(v is not None for v in vs) else None
            averaged.append(row)
        rows = averaged
    by_cell = defaultdict(dict)
    for r in rows:
        if _seed(r) in by_cell[_cell(r)]:
            raise ValueError("duplicate seed/cell in interaction analysis")
        by_cell[_cell(r)][_seed(r)] = r
    toolsets = sorted({_cell(r)[1] for r in rows})
    levels = sorted({float(r["filename_noise"]) for r in rows})
    if 0.0 not in levels or 0.9 not in levels:
        raise ValueError("interaction analysis requires noise 0 and 0.9")
    common = set.intersection(*(set(g) for g in by_cell.values()))
    if not common or any(set(g) != common for g in by_cell.values()):
        raise ValueError("interaction analysis requires the same seeds in all cells")
    within, between, interactions = [], [], []
    for metric in CONTRAST_METRICS:
        def values(cell, seeds):
            return [float(by_cell[cell][s][metric]) for s in seeds]
        def differences(left, right):
            seeds = sorted(s for s in common if by_cell[left][s].get(metric) is not None
                           and by_cell[right][s].get(metric) is not None)
            return [b - a for a, b in zip(values(left, seeds), values(right, seeds))]
        for ts in toolsets:
            within.append({"toolset": ts, "metric": metric, "contrast": "0.9 - 0",
                           **bootstrap_difference(differences((0, ts), (.9, ts)), seed=bootstrap_seed)})
        for a, b in combinations(toolsets, 2):
            for level in levels:
                between.append({"filename_noise": level, "toolset_a": a, "toolset_b": b,
                                "metric": metric, "contrast": f"{b} - {a}",
                                **bootstrap_difference(differences((level, a), (level, b)), seed=bootstrap_seed)})
            seeds = sorted(s for s in common if all(by_cell[(n, t)][s].get(metric) is not None
                                                   for n in (0, .9) for t in (a, b)))
            diffs = [(by_cell[(.9, b)][s][metric] - by_cell[(0, b)][s][metric])
                     - (by_cell[(.9, a)][s][metric] - by_cell[(0, a)][s][metric]) for s in seeds]
            interactions.append({"toolset_a": a, "toolset_b": b, "metric": metric,
                                 "contrast": f"noise sensitivity {b} - {a}",
                                 **bootstrap_difference(diffs, seed=bootstrap_seed)})
    failures = Counter((_cell(r)[1], r["filename_noise"], r.get("task_failure_type") or r.get("failure_type") or "success") for r in failure_rows)
    sensitivity = [{"toolset": ts, **{f"delta_{r['metric']}": r["mean_difference"]
                                    for r in within if r["toolset"] == ts}} for ts in toolsets]
    return {"summary": summarize([{**r, "toolset": _cell(r)[1]} for r in rows], ("toolset", "filename_noise")), "sensitivity": sensitivity, "within_toolset": within,
            "between_toolsets": between, "interaction": interactions,
            "failures": [{"toolset": t, "filename_noise": n, "failure_type": f, "n": count}
                         for (t, n, f), count in sorted(failures.items())]}


def write_interaction(rows: list[dict], out: str | Path, *, bootstrap_seed: int = 7) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = interaction_tables(rows, bootstrap_seed=bootstrap_seed)
    payload = {"bootstrap_seed": bootstrap_seed, "bootstrap_resamples": 10000, **tables}
    (out / "interaction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, records in tables.items():
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0]) if records else [])
            writer.writeheader()
            writer.writerows(records)
    lines = ["# Toolset and filename noise interaction", "",
             "Within-toolset paired changes: noise 0.9 minus noise 0. "
             "Intervals are percentile 95% bootstrap intervals across paired seeds "
             f"(10,000 resamples; seed {bootstrap_seed}).", "",
             "| Toolset | Delta calls | Delta files read | Delta tokens | Delta success |",
             "| --- | --- | --- | --- | --- |"]
    for r in tables["sensitivity"]:
        values = ["missing" if r[k] is None else format(r[k], spec) for k, spec in
                  (("delta_n_calls", "+.3f"), ("delta_files_read", "+.3f"),
                   ("delta_total_tokens", "+.1f"), ("delta_success", "+.3f"))]
        lines.append("| " + " | ".join([r["toolset"], *values]) + " |")
    lines += ["",
             "| Toolset | Metric | Pairs | Mean difference | 95% interval | Includes zero |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in tables["within_toolset"]:
        if r["n"]:
            lines.append(f"| {r['toolset']} | {r['metric']} | {r['n']} | {r['mean_difference']:.6g} | "
                         f"[{r['ci_low']:.6g}, {r['ci_high']:.6g}] | {r['includes_zero']} |")
    lines += ["", "At noise 0.9 (means across the same paired seeds):", "",
              "| Toolset | Success | Calls | Files read | Tokens | Cost (USD) | Wall time (s) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in tables["summary"]:
        if r["filename_noise"] == .9:
            def number(key, digits=3):
                return "missing" if r.get(key) is None else f"{r[key]:.{digits}f}"
            lines.append(f"| {r['toolset']} | {number('success')} | {number('n_calls')} | "
                         f"{number('files_read')} | {number('total_tokens', 1)} | "
                         f"{number('cost_usd', 6)} | {number('wall_time_s')} |")
    lines += ["", "Compare interfaces at each noise level in [between-toolset contrasts](between_toolsets.csv). "
              "The [interaction contrasts](interaction.csv) directly compare noise sensitivities with "
              "paired differences of differences. Positive values mean the second interface has a larger "
              "increase; the desirability depends on the metric.", "",
              "See [cell summaries](summary.csv) and [failure breakdown](failures.csv). "
              "Path hallucinations remain a separate rate; they are not a new correctness failure category.", "",
              "No combined score or automatic significance labels are used. Intervals are exploratory, "
              "unadjusted for multiple comparisons, and reflect seed variation rather than repeated "
              "model samples. A degenerate success interval from all-correct trials does not prove equal "
              "population accuracy. Null metrics are excluded pairwise; check each row's pair count.", ""]
    target = out / "interaction.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def compare_trace(root, seed: int, noise: float, *, model=None, trial_index=None) -> str:
    from fsbench.evaluate import find_runs, load_trace

    lines = [f"SEED {seed}  filename_noise={noise:g}"]
    found = []
    for run in find_runs(root):
        meta = json.loads((long_path(run) / "meta.json").read_text(encoding="utf-8"))
        if model is not None and meta.get("model") != model:
            continue
        if trial_index is not None and meta.get("trial_index", 1) != trial_index:
            continue
        if meta.get("world_seed") == seed and meta.get("filename_noise") == noise:
            found.append((_cell(meta)[1], run, meta))
    if not found:
        return "\n".join(lines + ["No matching runs."])
    identities = {(m.get("model"), m.get("task"), json.dumps(m.get("seeds"), sort_keys=True),
                   m.get("input_sha256"), m.get("experiment")) for _, _, m in found}
    if len(identities) != 1 or len({t for t, _, _ in found}) != len(found):
        raise ValueError("ambiguous or mismatched traces; select a single experiment/model/task run directory")
    for ts, run, meta in sorted(found):
        m = load_manifest(meta["env_dir"])
        files = {f["path"]: f for f in m["files"]}
        labels = {"required": "REQUIRED EVIDENCE", "trap": "STALE/DRAFT TRAP",
                  "near": "DISTRACTOR", "generic": "DISTRACTOR"}
        lines += ["", ts.upper()]
        first = None
        for e in load_trace(long_path(run) / "trace.jsonl"):
            notes = []
            for p in e["read"]:
                if p in files:
                    notes.append(f"{labels[files[p]['role']]}: {p}")
                    if files[p]["role"] == "required" and first is None:
                        first = e["step"]
            notes += [f"INVALID PATH: {p}" for p in e["missing"]]
            if not e["ok"]:
                notes.append(f"ERROR: {e['error']}")
            lines.append(f"{e['step']}. {e['tool']}({json.dumps(e['args'], sort_keys=True)})"
                         + ("  [" + "; ".join(notes) + "]" if notes else ""))
        lines.append(f"Required evidence at step {first}" if first else "Required evidence never read")
    return "\n".join(lines)
