"""CLI commands for candidate packs, human collection and observed calibration."""
import json
from pathlib import Path


def build(a):
    from fsbench.taskpacks import build_pack
    from fsbench.validity import benchmark_card
    excluded = a.exclude_benchmark
    if a.cmd=="build-testset" and not excluded and Path("datasets/realistic-v1/benchmark.json").exists():
        excluded = "datasets/realistic-v1"
    data = build_pack(a.out,version=a.task_pack,num_worlds=a.num_worlds,seed_file=a.seed_file,test_only=a.cmd=="build-testset",exclude_root=excluded)
    benchmark_card(a.out)
    print(f"Built candidate {data['benchmark_version']}: {data['worlds']} worlds, {data['tasks']} tasks. Human validation pending.")


def validate(a):
    from fsbench.validity import validate_benchmark, benchmark_card
    root = a.root or "datasets/"+a.benchmark
    report = validate_benchmark(root,a.benchmark)
    if (Path(root)/"benchmark.json").exists():
        benchmark_card(root)
    for label,key in (("Tasks","tasks"),("Worlds","worlds"),("Solvable","solvable"),("Oracle success","oracle_success"),
                      ("Answer leakage","answer_leakage"),("Missing evidence","missing_evidence"),("Parser errors","parser_errors")):
        print(f"{label}: {report[key]}")
    print("AUTOMATED VALID; human validation pending" if report["automated_valid"] else "INVALID")
    print("Release gates: "+json.dumps(report["release_criteria"]))
    if not report["automated_valid"]:
        raise SystemExit(1)


def card(a):
    from fsbench.validity import benchmark_card
    print(benchmark_card(a.root,out=a.out,human_runs=a.human_runs))


def human(a):
    from fsbench.human import human_run
    human_run(a.env,a.participant,a.out,interface=a.interface,toolset=a.toolset)


def calibration(a):
    from fsbench.calibration import calibrate
    data = calibrate(a.runs,a.human_runs,a.out,min_humans=a.min_humans)
    print(f"Wrote {len(data['tasks'])} observed task summaries to {a.out}; no observations are fabricated.")


def study(a):
    """Select one task per requested family per world, with no answer-key fields."""
    from fsbench.evaluate import find_envs
    from fsbench.benchmark import load_manifest
    from collections import defaultdict
    if a.planned_models<1 or len(a.model or [])>a.planned_models:
        raise ValueError("planned model count must be positive and include every supplied model")
    grouped = defaultdict(list)
    for env in find_envs(a.root):
        m = load_manifest(env)
        grouped[m.get("world_id")].append((env,m))
    chosen = []
    for world,items in sorted(grouped.items()):
        used, selected = set(),[]
        for family in a.families:
            possible = [(e,m) for e,m in items if family in m.get("task_families",[]) and m["env_id"] not in used]
            if not possible:
                break
            env,m = sorted(possible,key=lambda pair:pair[1]["task_template"])[0]
            used.add(m["env_id"])
            selected.append({"env":str(env.resolve()),"instance_id":m["env_id"],"world_id":world,
                             "task_family":family,"domain":m["workspace_domain"],"split":m["split"]})
        if len(selected)==len(a.families):
            chosen.append(selected)
    # Round-robin domains avoids selecting 20 worlds from a single domain.
    buckets = defaultdict(list)
    for selected in chosen:
        buckets[selected[0]["domain"]].append(selected)
    selected_worlds = []
    while len(selected_worlds)<a.worlds and any(buckets.values()):
        for domain in sorted(buckets):
            if buckets[domain] and len(selected_worlds)<a.worlds:
                selected_worlds.append(buckets[domain].pop(0))
    if len(selected_worlds)<a.worlds:
        raise ValueError("not enough worlds with distinct tasks for every requested family")
    payload = {"status":"prepared; no model or human observations collected","models":a.model or [],
        "families":a.families,"worlds":a.worlds,"instances":[r for rows in selected_worlds for r in rows],
        "planned_model_count":a.planned_models,"model_choices_pending":a.planned_models-len(a.model or []),
        "expected_model_runs":a.worlds*len(a.families)*a.planned_models,
        "note":"Supply three model IDs before running; use voluntary participants for human baselines. Do not treat sibling tasks as independent worlds."}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Prepared {len(payload['instances'])} tasks from {a.worlds} worlds in {a.out}")


def add_commands(sub):
    for name in ("build-benchmark","build-testset"):
        p = sub.add_parser(name,help="build a version-locked realistic candidate pack")
        p.add_argument("--task-pack",default="realistic-v1")
        p.add_argument("--num-worlds",type=int,default=100)
        p.add_argument("--seed-file",required=name=="build-testset")
        p.add_argument("--exclude-benchmark",help="reject world overlap with an existing benchmark (testsets default to local realistic-v1 if present)")
        p.add_argument("--out",default="datasets/realistic-v1" if name=="build-benchmark" else "datasets/heldout-realistic-v1")
        p.set_defaults(fn=build)
    p = sub.add_parser("validate-benchmark",help="check solvability, evidence, parsers, leakage and split integrity")
    p.add_argument("--benchmark",default="realistic-v1")
    p.add_argument("--root")
    p.set_defaults(fn=validate)
    p = sub.add_parser("benchmark-card",help="write an honest benchmark card with measured validation status")
    p.add_argument("--root",default="datasets/realistic-v1")
    p.add_argument("--out")
    p.add_argument("--human-runs",help="include counts of supplied baseline observations for this benchmark version")
    p.set_defaults(fn=card)
    p = sub.add_parser("human-run",help="collect a voluntary pseudonymous human baseline")
    p.add_argument("--env",required=True)
    p.add_argument("--participant",required=True)
    p.add_argument("--out")
    p.add_argument("--interface",choices=["cli","native"],default="cli")
    p.add_argument("--toolset",choices=["files","shell","indexed"],default="files")
    p.set_defaults(fn=human)
    p = sub.add_parser("calibrate",help="summarize observed human and agent difficulty without a combined score")
    p.add_argument("--runs")
    p.add_argument("--human-runs",default="runs/human")
    p.add_argument("--out",default="runs/calibration")
    p.add_argument("--min-humans",type=int,default=3)
    p.set_defaults(fn=calibration)
    p = sub.add_parser("prepare-study",help="select a balanced realistic validation study without model calls")
    p.add_argument("--root",default="datasets/realistic-v1/validation")
    p.add_argument("--worlds",type=int,default=20)
    p.add_argument("--families",nargs="+",default=["source_selection","provenance","multi_file_reasoning"])
    p.add_argument("--model",action="append")
    p.add_argument("--planned-models",type=int,default=3)
    p.add_argument("--out",default="runs/realistic-study/study.json")
    p.set_defaults(fn=study)
