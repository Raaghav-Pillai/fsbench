"""Command-line entry point: ``fsbench <command> ...`` (or ``python -m fsbench``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fsbench.config import PRESETS, FSConfig
from fsbench.benchmark import load_manifest, require_evaluator
from fsbench.evaluate import (
    SUMMARY_METRICS,
    evaluate_runs,
    find_envs,
    mixed_version_warnings,
    select_envs,
    summarize,
)
from fsbench.generate import generate_env
from fsbench.paths import long_path
from fsbench.runner import OracleAgent, run_agent
from fsbench.sweep import run_sweep
from fsbench.tasks import TASK_TYPES
from fsbench.tools import TOOLSETS, make_toolset, render_context


def parse_scalar(s: str):
    s = s.strip()
    if s.lower() in ("none", "null"):
        return None
    if "+" in s:
        return tuple(x for x in s.split("+") if x)
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _kv(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v


def build_config(preset: str | None, sets: list[tuple[str, str]]) -> FSConfig:
    cfg = PRESETS[preset] if preset else FSConfig()
    for k, v in sets:
        val = parse_scalar(v)
        if k == "mime_types" and isinstance(val, str):
            val = (val,)
        cfg = cfg.with_value(k, val)
    return cfg


def _config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", "--profile", choices=sorted(PRESETS), help="start from a named condition")
    p.add_argument("--set", dest="sets", type=_kv, action="append", default=[], metavar="KEY=VALUE",
                   help="override a knob, e.g. --set depth=6 --set mime_types=txt+pdf+xlsx "
                        "--set mime_diversity=4. Repeatable.")


def _generate_label(preset: str | None, sets: list[tuple[str, str]], label: str | None) -> str:
    if label:
        return label
    extras = ",".join(f"{k}={v}" for k, v in sets)
    if preset and extras:
        return f"{preset}:{extras}"
    if preset:
        return preset
    if len(sets) == 1 and sets[0][0] == "filename_noise":
        return f"filename_noise={float(sets[0][1]):g}"
    return f"custom:{extras}" if extras else "default"


def cmd_generate(a) -> None:
    cfg = build_config(a.preset, a.sets)
    ws, ts, ls = (a.seed if s is None else s for s in (a.world_seed, a.task_seed, a.layout_seed))
    m = generate_env(a.task, cfg, a.out, world_seed=ws, task_seed=ts, layout_seed=ls,
                     label=_generate_label(a.preset, a.sets, a.label))
    print(f"Generated {m['env_id']}: {m['stats']['n_files']} files in {m['stats']['n_dirs']} folders")
    print(f"  workspace: {Path(a.out) / 'workspace'}")
    print(f"  task:      {Path(a.out) / 'task.json'}")
    print(f"  manifest:  {Path(a.out) / 'manifest.json'}  (hidden from the agent)")


def cmd_sweep(a) -> None:
    if a.profiles:
        from fsbench.sweep import run_profiles
        if a.vary or a.preset:
            sys.exit("--profiles compares named conditions; use --profile with --vary for a factorial sweep")
        overrides = [(k, parse_scalar(v)) for k, v in a.sets]
        overrides = [(k, (v,) if k == "mime_types" and isinstance(v, str) else v) for k, v in overrides]
        envs = run_profiles(a.task, a.profiles, a.out, replicates=a.replicates,
                            seed_offset=a.seed_offset, overrides=overrides)
        print(f"Generated and verified {len(envs)} paired profile environments under {a.out}")
        return
    base = build_config(a.preset, a.sets)
    vary = {}
    for k, v in a.vary:
        vary[k] = [parse_scalar(x) for x in v.split(",")]
    if not vary:
        sys.exit("at least one --vary KEY=V1,V2,... is required")

    def progress(i, n, env_id):
        if i == n or i % max(1, n // 20) == 0:
            print(f"  [{i}/{n}] {env_id}", flush=True)

    index = run_sweep(a.task, base, vary, a.out, replicates=a.replicates,
                      mode="ofat" if a.ofat else "grid", seed_offset=a.seed_offset, progress=progress)
    print(f"Generated {len(index)} environments under {a.out} (see index.csv)")


def cmd_oracle(a) -> None:
    envs = find_envs(a.envs)
    if not envs:
        sys.exit(f"no environments found under {a.envs}")
    root = Path(a.envs).resolve()
    ok = 0
    for env in envs:
        manifest = load_manifest(env)
        require_evaluator(manifest, a.evaluator)
        run_name = "__".join(env.resolve().relative_to(root).parts) or env.name
        for ts in a.toolset:
            m = run_agent(env, OracleAgent(manifest), Path(a.out) / ts / run_name, toolset=ts)
            ok += m["success"]
            if not m["success"]:
                print(f"  FAIL {env.name} [{ts}]: {m['outcome']}")
    total = len(envs) * len(a.toolset)
    print(f"Oracle solved {ok}/{total} runs; results under {a.out}")


FATAL_API_ERRORS = ("HTTP 401", "HTTP 402", "HTTP 403", "No OpenRouter API key")


def cmd_run(a) -> None:
    from fsbench.models import create_agent
    from fsbench.experiment import build_schedule, save_plan, validate_environment, fingerprint
    from fsbench.policies import validate_policy
    from fsbench.provenance import implementation_fingerprint

    envs = select_envs(find_envs(a.envs), replicates=a.replicates, only=a.only)
    study_families = {}
    if a.study:
        study = json.loads(Path(a.study).read_text(encoding="utf-8"))
        wanted = {Path(r["env"]).resolve() for r in study["instances"]}
        available = {e.resolve() for e in envs}
        if not wanted <= available:
            sys.exit("study contains environments outside --envs or excluded by the run filters")
        envs = [e for e in envs if e.resolve() in wanted]
        study_families = {r["instance_id"]:r["task_family"] for r in study["instances"]}
    if not envs:
        sys.exit(f"no environments found under {a.envs} (check --replicates / --only)")
    try:
        agents = {model: create_agent(a.provider, model, max_steps=a.max_steps, temperature=a.temperature)
                  for model in sorted(set(a.model))}
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    root = Path(a.envs).resolve()
    arms = [tuple(v.split(":")) for v in a.arm] if a.arm else [(t, a.tool_policy) for t in a.toolset]
    if any(len(arm) != 2 for arm in arms):
        sys.exit("--arm must be TOOLSET:POLICY")
    for ts, policy in arms:
        if ts not in TOOLSETS:
            sys.exit(f"unknown toolset: {ts}")
        validate_policy(ts, policy)
    model_dirs = {m: m.replace("/", "__").replace(":", "_") for m in agents}
    if len(set(model_dirs.values())) != len(model_dirs):
        sys.exit("model names map to the same output directory; choose distinct model identifiers")
    schedule = build_schedule(envs, a.toolset, order_seed=a.order_seed,
                              check_index=any(t == "indexed" for t, _ in arms), arms=arms,
                              models=list(agents), trials_per_cell=a.trials_per_cell)
    for cell in schedule:
        if cell["env_id"] in study_families:
            cell["study_family"] = study_families[cell["env_id"]]
    experiment = a.experiment or ("filename_noise_toolset_interaction_v1" if len(set(a.toolset)) > 1 else "fsbench")
    dimensions = ("task", "condition_id", "model", "toolset", "tool_policy", "trial_index")
    expected_by_key = {}
    for cell in schedule:
        expected = {k:cell[k] for k in dimensions}
        if cell.get("benchmark_family") == "realistic":
            expected.update({k:cell[k] for k in ("world_seed","task_seed","layout_seed")})
        expected_by_key[json.dumps(expected,sort_keys=True)] = expected
    expected_cells = [expected_by_key[k] for k in sorted(expected_by_key)]
    plan = {"design_version": 2, "experiment": experiment, "models": sorted(agents),
            "implementation_sha256": implementation_fingerprint(),
            "provider": a.provider, "temperature": a.temperature,
            "max_steps": a.max_steps, "allow_writes": not a.no_writes,
            "experiment_order_seed": a.order_seed, "expected_cells": expected_cells,
            "agent_describe": {model: agent.describe() for model, agent in agents.items()}, "schedule": schedule}
    plan["design_id"] = fingerprint(plan)
    save_plan(Path(a.out) / "experiment.json", plan)
    total_cost, done, solved, unreported_cost = 0.0, 0, 0, 0
    for cell in schedule:
        env, ts = Path(cell["env_dir"]), cell["toolset"]
        agent = agents[cell["model"]]
        model_dir = model_dirs[cell["model"]]
        run_name = "__".join(env.resolve().relative_to(root).parts) or env.name
        run_dir = Path(a.out) / model_dir / ts / run_name
        if cell["tool_policy"] != "naturalistic":
            run_dir = Path(a.out) / model_dir / (ts + "__" + cell["tool_policy"]) / run_name
        if a.trials_per_cell > 1:
            run_dir = run_dir / f"trial_{cell['trial_index']:03d}"
        if a.skip_existing and (long_path(run_dir) / "metrics.json").exists():
            meta = json.loads((long_path(run_dir) / "meta.json").read_text(encoding="utf-8"))
            if any(meta.get(k) != v for k, v in {"experiment": experiment,
                    "execution_order": cell["execution_order"], "input_sha256": cell["input_sha256"],
                    "design_id": plan["design_id"], "trial_index": cell["trial_index"],
                    "experiment_order_seed": a.order_seed, "agent_describe": agent.describe()}.items()):
                sys.exit(f"existing run does not match experiment plan: {run_dir}")
            continue
        if validate_environment(env) != {k: cell[k] for k in ("workspace_sha256", "input_sha256")}:
            sys.exit(f"environment changed since planning: {env}")
        context = {**cell, "experiment": experiment, "expected_cells": expected_cells,
                   "design_version": 2, "design_id": plan["design_id"], "provider": a.provider,
                   "implementation_sha256": plan["implementation_sha256"]}
        m = run_agent(env, agent, run_dir, toolset=ts, tool_policy=cell["tool_policy"],
                      allow_writes=not a.no_writes, run_context=context)
        meta = json.loads((long_path(run_dir) / "meta.json").read_text(encoding="utf-8"))
        cost = (m.get("usage") or {}).get("cost_usd")
        total_cost += cost or 0
        unreported_cost += cost is None
        cost_label = f"${cost:.6f}" if cost is not None else "unreported"
        done += 1
        solved += m["success"]
        status = "PASS" if m["success"] else "FAIL"
        print(f"  [{cell['execution_order']}/{len(schedule)}] {status} {run_name} [{ts}/{cell['tool_policy']}] "
              f"score={m['score']:.2f} calls={m['n_calls']} cost={cost_label}", flush=True)
        err = meta.get("agent_error") or ""
        if err:
            print("    agent error: " + err.strip().splitlines()[-1])
            if any(s in err for s in FATAL_API_ERRORS):
                sys.exit("Stopping: the API rejected the request (check your key, credits, or model access).")
    print(f"{solved}/{done} runs succeeded; reported cost ${total_cost:.4f}; "
          f"{unreported_cost} runs lack cost data. Results under {a.out}")
    print(f"Summarise with: fsbench evaluate --runs {a.out} --out results.csv")


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    fmt = [[("" if r[c] is None else format(r[c], ".6f" if c == "cost_usd" else ".3f")
             if isinstance(r[c], float) else str(r[c])) for c in cols]
           for r in rows]
    widths = [max(len(c), *(len(f[i]) for f in fmt)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for f in fmt:
        print("  ".join(x.ljust(w) for x, w in zip(f, widths)))


def cmd_evaluate(a) -> None:
    expected = {k: v.split(",") for k, v in a.expect}
    rows = evaluate_runs(a.runs, a.out, paired=a.paired, expected_values=expected)
    if not rows:
        sys.exit(f"no runs available under {a.runs} after any requested pairing")
    for w in mixed_version_warnings(rows):
        print(f"warning: {w}", file=sys.stderr)
    print(f"Evaluated {len(rows)} runs" + (f"; wrote {a.out}" if a.out else ""))
    by = tuple(a.group_by.split(",")) if a.group_by else ("agent", "toolset", "tool_policy", "task_type", "condition")
    _print_table(summarize(rows, by))
    if a.interaction:
        from fsbench.interaction import write_interaction
        target = Path(a.out).parent if a.out else Path(a.runs)
        report = write_interaction(rows, target, bootstrap_seed=a.bootstrap_seed)
        print(f"Interaction analysis: {report}")
    if a.statistics:
        from fsbench.statistics import write_statistics
        target = Path(a.out).parent if a.out else Path(a.runs)
        print(f"Statistics: {write_statistics(rows, target, baseline=a.baseline, bootstrap_seed=a.bootstrap_seed)}")


def cmd_validate(a) -> None:
    from fsbench.experiment import validate_environment
    from fsbench.integrity import verify

    if a.filename_noise:
        print(json.dumps(verify(a.envs), indent=2))
    envs = find_envs(a.envs)
    if not envs:
        sys.exit("no environments found")
    if a.composed:
        from fsbench.difficulty import verify_composed
        print(json.dumps(verify_composed(envs), indent=2))
    for env in envs:
        validate_environment(env, check_index=a.index)
    print(f"Validated {len(envs)} environments" + (" and their indexes" if a.index else ""))


def cmd_compare_trace(a) -> None:
    from fsbench.interaction import compare_trace
    print(compare_trace(a.runs, a.seed, a.noise, model=a.model, trial_index=a.trial_index))


def cmd_plot(a) -> None:
    from fsbench.plot import plot_runs

    rows = evaluate_runs(a.runs, paired=a.paired,
                         expected_values={k: v.split(",") for k, v in a.expect})
    if not rows:
        sys.exit(f"no runs available under {a.runs} after any requested pairing")
    try:
        paths = plot_runs(rows, a.x, a.out)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"Wrote {len(paths)} plots under {a.out}")
    for p in paths:
        print(f"  {p}")


def cmd_context(a) -> None:
    sys.stdout.write(render_context(long_path(a.env) / "workspace", include_paths=not a.no_paths))


def cmd_tools(a) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(make_toolset(a.toolset, d, allow_writes=not a.no_writes).specs(a.style), indent=2))


def cmd_inspect(a) -> None:
    if a.run:
        from fsbench.behavior import inspect_run
        from fsbench.evaluate import find_runs
        run = Path(a.run)
        if not (run / "meta.json").is_file():
            matches = [p for p in find_runs(a.runs) if p.name == a.run]
            if len(matches) != 1:
                sys.exit(f"run ID matched {len(matches)} runs; pass a unique run directory")
            run = matches[0]
        meta = json.loads((long_path(run)/"meta.json").read_text(encoding="utf-8"))
        require_evaluator(load_manifest(meta["env_dir"]),a.evaluator)
        print(inspect_run(run))
        return
    m = load_manifest(a.env)
    require_evaluator(m,a.evaluator)
    print(f"{m['env_id']}  task={m['task']['type']} (level {m['task']['level']})  seeds={m['seeds']}")
    print(f"config: {json.dumps(m['config'])}")
    print(f"stats:  {json.dumps(m['stats'])}")
    print(f"oracle: {json.dumps(m['oracle'])}")
    print(f"ground truth: {json.dumps(m['ground_truth'])}")
    roles = ("required", "trap") if not a.all else ("required", "trap", "near", "generic")
    for f in m["files"]:
        if f["role"] in roles:
            print(f"  [{f['role']:>8}] {f['path']}   ({f['doc_id']}, {f['format']})")


def cmd_behavior(a):
    from fsbench.behavior import analyze_behavior
    from fsbench.statistics import write_csv
    rows = evaluate_runs(a.runs, paired=a.paired)
    if not rows:
        sys.exit("no runs available")
    records = analyze_behavior(rows, a.group_by)
    out = Path(a.out) if a.out else Path(a.runs) / "behavior.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, records)
    print(f"Analyzed {len(rows)} runs; wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fsbench", description="FS-EntropyBench environment generator and evaluator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="build one environment")
    p.add_argument("--task", required=True, choices=sorted(TASK_TYPES))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="default for world/task/layout seeds")
    p.add_argument("--world-seed", type=int)
    p.add_argument("--task-seed", type=int)
    p.add_argument("--layout-seed", type=int)
    p.add_argument("--label", help="condition label stored in the manifest (default: preset or custom:overrides)")
    _config_args(p)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("sweep", help="build an experiment matrix")
    p.add_argument("--task", required=True, nargs="+", choices=sorted(TASK_TYPES))
    p.add_argument("--out", required=True)
    p.add_argument("--vary", type=_kv, action="append", default=[], metavar="KEY=V1,V2,...")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    sweep_mode = p.add_mutually_exclusive_group()
    sweep_mode.add_argument("--grid", action="store_true", help="full factorial (default)")
    sweep_mode.add_argument("--ofat", action="store_true", help="vary one factor at a time")
    p.add_argument("--profiles", nargs="+", choices=sorted(PRESETS), help="compare named profiles on paired worlds")
    _config_args(p)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("oracle", help="run the oracle agent over environments (sanity check)")
    p.add_argument("--evaluator",action="store_true",help="explicit access to held-out evaluator labels")
    p.add_argument("--envs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--toolset", nargs="+", default=["files"], choices=sorted(TOOLSETS))
    p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("run", help="run models and repeated trials across tool policies")
    p.add_argument("--envs", required=True, help="an environment folder or a folder of environments")
    p.add_argument("--study",help="filter to the exact environments in a prepared study JSON")
    p.add_argument("--model", required=True, action="append", help="repeat for each model to compare")
    p.add_argument("--provider", default="openrouter", help="registered adapter provider (openrouter or compatible)")
    p.add_argument("--trials-per-cell", type=int, default=1)
    p.add_argument("--interleave", action="store_true", help="interleaving is always enabled")
    p.add_argument("--out", required=True)
    p.add_argument("--toolset", nargs="+", default=["files"], choices=sorted(TOOLSETS))
    p.add_argument("--max-steps", type=int, default=40, help="maximum model turns per run")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--replicates", type=int, help="keep environments with sweep.replicate < N (balanced seeds)")
    p.add_argument("--only", help="keep environments from one swept variable, e.g. depth")
    p.add_argument("--no-writes", action="store_true", help="disable write_file (scratch-memory ablation)")
    p.add_argument("--skip-existing", action="store_true", help="skip runs that already have metrics.json")
    p.add_argument("--order-seed", type=int, default=7, help="reproducible within-seed trial order")
    p.add_argument("--experiment", help="experiment identifier recorded with the saved schedule")
    from fsbench.policies import POLICIES
    p.add_argument("--tool-policy", choices=POLICIES, default="naturalistic")
    p.add_argument("--arm", action="append", default=[], metavar="TOOLSET:POLICY",
                   help="repeat for specific interface/policy pairs; overrides --toolset/--tool-policy")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("evaluate", help="score runs and summarise")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", help="write per-run CSV here")
    p.add_argument("--paired", action="store_true", help="drop seeds that are missing from any condition")
    p.add_argument("--expect", type=_kv, action="append", default=[], metavar="KEY=V1,V2,...",
                   help="expected conditions (new sweeps also store these automatically)")
    p.add_argument("--group-by", help=f"comma-separated columns (default agent,toolset,task_type,condition); "
                                      f"summary metrics: {', '.join(SUMMARY_METRICS)}")
    p.add_argument("--interaction", action="store_true", help="write paired toolset/noise tables and bootstrap intervals")
    p.add_argument("--bootstrap-seed", type=int, default=7)
    p.add_argument("--statistics", action="store_true", help="world-clustered CIs, robustness/model contrasts, variation and factorial interactions")
    p.add_argument("--baseline", default="clean", help="condition label for stress minus baseline contrasts")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("plot", help="plot success/calls/tokens/cost/latency against one variable")
    p.add_argument("--runs", required=True)
    p.add_argument("--x", required=True, help="filesystem variable, e.g. depth or filename_noise")
    p.add_argument("--out", required=True, help="directory for PNG files")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--expect", type=_kv, action="append", default=[], metavar="KEY=V1,V2,...")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("validate", help="check visible file bytes and optional filename pairing/index integrity")
    p.add_argument("--envs", required=True)
    p.add_argument("--filename-noise", action="store_true")
    p.add_argument("--composed", action="store_true", help="verify stable facts and required evidence across profiles")
    p.add_argument("--index", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("compare-trace", help="annotate matching trials across toolsets")
    p.add_argument("--runs", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--model")
    p.add_argument("--trial-index", type=int)
    p.set_defaults(fn=cmd_compare_trace)

    p = sub.add_parser("context", help="dump every file as one text block (no-filesystem condition)")
    p.add_argument("--env", required=True)
    p.add_argument("--no-paths", action="store_true")
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("tools", help="print tool schemas for a toolset")
    p.add_argument("--toolset", required=True, choices=sorted(TOOLSETS))
    p.add_argument("--style", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--no-writes", action="store_true")
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("inspect", help="show an environment manifest or a run's annotated trajectory")
    p.add_argument("--evaluator",action="store_true",help="explicit access to held-out evaluator labels")
    inspect_target = p.add_mutually_exclusive_group(required=True)
    inspect_target.add_argument("--env")
    inspect_target.add_argument("--run", help="run directory or unique folder name")
    p.add_argument("--runs", default="runs", help="root to search for a run ID")
    p.add_argument("--all", action="store_true", help="also list distractor files")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("analyze-behavior", help="group deterministic trajectory metrics and failure classes")
    p.add_argument("--runs", required=True)
    p.add_argument("--out")
    p.add_argument("--group-by", nargs="+", default=["model", "filename_noise", "toolset", "tool_policy"])
    p.add_argument("--paired", action="store_true")
    p.set_defaults(fn=cmd_behavior)

    from fsbench.benchmark_cli import add_commands
    add_commands(sub)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
