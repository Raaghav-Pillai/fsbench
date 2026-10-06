"""Command-line entry point: ``fsbench <command> ...`` (or ``python -m fsbench``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fsbench.config import PRESETS, FSConfig
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
    p.add_argument("--preset", choices=sorted(PRESETS), help="start from a named condition")
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
                      mode="grid" if a.grid else "ofat", seed_offset=a.seed_offset, progress=progress)
    print(f"Generated {len(index)} environments under {a.out} (see index.csv)")


def cmd_oracle(a) -> None:
    envs = find_envs(a.envs)
    if not envs:
        sys.exit(f"no environments found under {a.envs}")
    root = Path(a.envs).resolve()
    ok = 0
    for env in envs:
        manifest = json.loads((long_path(env) / "manifest.json").read_text(encoding="utf-8"))
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
    from fsbench.openrouter import OpenRouterAgent

    envs = select_envs(find_envs(a.envs), replicates=a.replicates, only=a.only)
    if not envs:
        sys.exit(f"no environments found under {a.envs} (check --replicates / --only)")
    try:
        agent = OpenRouterAgent(a.model, max_steps=a.max_steps, temperature=a.temperature)
    except RuntimeError as e:
        sys.exit(str(e))
    root = Path(a.envs).resolve()
    model_dir = a.model.replace("/", "__").replace(":", "_")
    total_cost, done, solved = 0.0, 0, 0
    for env in envs:
        run_name = "__".join(env.resolve().relative_to(root).parts) or env.name
        for ts in a.toolset:
            run_dir = Path(a.out) / model_dir / ts / run_name
            if a.skip_existing and (long_path(run_dir) / "metrics.json").exists():
                continue
            m = run_agent(env, agent, run_dir, toolset=ts, allow_writes=not a.no_writes)
            meta = json.loads((long_path(run_dir) / "meta.json").read_text(encoding="utf-8"))
            cost = (m.get("usage") or {}).get("cost_usd") or 0.0
            total_cost += cost
            done += 1
            solved += m["success"]
            status = "PASS" if m["success"] else "FAIL"
            print(f"  {status} {run_name} [{ts}] score={m['score']:.2f} calls={m['n_calls']} "
                  f"discovery_regret={m['discovery_regret']} cost=${cost:.4f}", flush=True)
            err = meta.get("agent_error") or ""
            if err:
                print("    agent error: " + err.strip().splitlines()[-1])
                if any(s in err for s in FATAL_API_ERRORS):
                    sys.exit("Stopping: the API rejected the request (check your key, credits, or model access).")
    print(f"{solved}/{done} runs succeeded; total cost ${total_cost:.4f}. Results under {a.out}")
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
    by = tuple(a.group_by.split(",")) if a.group_by else ("agent", "toolset", "task_type", "condition")
    _print_table(summarize(rows, by))


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
    m = json.loads((long_path(a.env) / "manifest.json").read_text(encoding="utf-8"))
    print(f"{m['env_id']}  task={m['task']['type']} (level {m['task']['level']})  seeds={m['seeds']}")
    print(f"config: {json.dumps(m['config'])}")
    print(f"stats:  {json.dumps(m['stats'])}")
    print(f"oracle: {json.dumps(m['oracle'])}")
    print(f"ground truth: {json.dumps(m['ground_truth'])}")
    roles = ("required", "trap") if not a.all else ("required", "trap", "near", "generic")
    for f in m["files"]:
        if f["role"] in roles:
            print(f"  [{f['role']:>8}] {f['path']}   ({f['doc_id']}, {f['format']})")


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
    p.add_argument("--grid", action="store_true", help="full factorial instead of one-factor-at-a-time")
    _config_args(p)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("oracle", help="run the oracle agent over environments (sanity check)")
    p.add_argument("--envs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--toolset", nargs="+", default=["files"], choices=sorted(TOOLSETS))
    p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("run", help="run an OpenRouter model as the agent over environments")
    p.add_argument("--envs", required=True, help="an environment folder or a folder of environments")
    p.add_argument("--model", required=True, help="OpenRouter model slug, e.g. openai/gpt-4o-mini")
    p.add_argument("--out", required=True)
    p.add_argument("--toolset", nargs="+", default=["files"], choices=sorted(TOOLSETS))
    p.add_argument("--max-steps", type=int, default=40, help="maximum model turns per run")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--replicates", type=int, help="keep environments with sweep.replicate < N (balanced seeds)")
    p.add_argument("--only", help="keep environments from one swept variable, e.g. depth")
    p.add_argument("--no-writes", action="store_true", help="disable write_file (scratch-memory ablation)")
    p.add_argument("--skip-existing", action="store_true", help="skip runs that already have metrics.json")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("evaluate", help="score runs and summarise")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", help="write per-run CSV here")
    p.add_argument("--paired", action="store_true", help="drop seeds that are missing from any condition")
    p.add_argument("--expect", type=_kv, action="append", default=[], metavar="KEY=V1,V2,...",
                   help="expected conditions (new sweeps also store these automatically)")
    p.add_argument("--group-by", help=f"comma-separated columns (default agent,toolset,task_type,condition); "
                                      f"summary metrics: {', '.join(SUMMARY_METRICS)}")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("plot", help="plot success/calls/tokens/cost/latency against one variable")
    p.add_argument("--runs", required=True)
    p.add_argument("--x", required=True, help="filesystem variable, e.g. depth or filename_noise")
    p.add_argument("--out", required=True, help="directory for PNG files")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--expect", type=_kv, action="append", default=[], metavar="KEY=V1,V2,...")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("context", help="dump every file as one text block (no-filesystem condition)")
    p.add_argument("--env", required=True)
    p.add_argument("--no-paths", action="store_true")
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("tools", help="print tool schemas for a toolset")
    p.add_argument("--toolset", required=True, choices=sorted(TOOLSETS))
    p.add_argument("--style", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--no-writes", action="store_true")
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("inspect", help="show an environment's manifest summary")
    p.add_argument("--env", required=True)
    p.add_argument("--all", action="store_true", help="also list distractor files")
    p.set_defaults(fn=cmd_inspect)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
