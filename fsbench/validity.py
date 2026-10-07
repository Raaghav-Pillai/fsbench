"""Automated solvability/leakage checks and an explicit, limited benchmark card."""
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from fsbench.benchmark import load_manifest
from fsbench.citations import score_citations
from fsbench.evaluate import find_envs
from fsbench.experiment import validate_environment, fingerprint, workspace_inventory
from fsbench.extract import extract_text
from fsbench.realistic import DOMAINS, LEVELS, materialize_world, score_realistic
from fsbench.realistic_oracle import solve
from fsbench.taskpacks import check_version
from fsbench.tools import make_toolset

FORBIDDEN = re.compile(r"ground[_ -]?truth|required[_ -]?(?:doc(?:ument)?|file|evidence)|is[_ -]?answer|relevance[_ -]?(?:label|score)|generation_seed|decoy_answers|answer[_ -]?key|manifest\.json|\"answer\"\s*:", re.I)


def leakage_findings(workspace, cache=None):
    cache = cache if cache is not None else {}
    issues = []
    for path in sorted(Path(workspace).rglob("*")):
        rel = path.relative_to(workspace).as_posix()
        if FORBIDDEN.search(rel):
            issues.append({"path":rel,"kind":"filename_metadata_leak"})
        if path.is_symlink() or (hasattr(path,"is_junction") and path.is_junction()):
            issues.append({"path":rel,"kind":"workspace_link"})
            continue
        if not path.is_file():
            continue
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            if digest not in cache:
                cache[digest] = extract_text(path)
            text = cache[digest]
            if not text.strip():
                issues.append({"path":rel,"kind":"empty_document"})
            if FORBIDDEN.search(text):
                issues.append({"path":rel,"kind":"content_metadata_leak"})
        except Exception as e:
            issues.append({"path":rel,"kind":"parser_error","error":str(e)[:200]})
    return issues


def validate_benchmark(root, version="realistic-v1", *, out=None):
    check_version(version)
    envs = find_envs(root)
    if not envs:
        raise ValueError("no benchmark tasks found; build the pack first")
    errors, cache, checked_worlds = [], {}, set()
    solvable, oracle_ok, missing, parser_errors, leakage = 0,0,0,0,0
    domains, templates, levels, world_splits = set(),defaultdict(set),set(),defaultdict(set)
    invariant_worlds = set()
    for env in envs:
        m = load_manifest(env)
        instance = m["env_id"]
        try:
            if m.get("benchmark_version") != version or not m.get("ground_truth"):
                raise ValueError("missing ground truth or benchmark version mismatch")
            validate_environment(env)
            required = set(m["required_doc_ids"])
            if not required or not required <= {f["doc_id"] for f in m["files"]}:
                missing += 1
                raise ValueError("missing required evidence")
            domains.add(m["workspace_domain"])
            templates[m["workspace_domain"]].add(m["task_template"])
            levels.add(m["difficulty_level"])
            world_splits[m["world_id"]].add(m["split"])
            if m["world_id"] not in checked_worlds:
                checked_worlds.add(m["world_id"])
                findings = leakage_findings(env/"workspace",cache)
                parser_errors += sum(f["kind"] in ("parser_error","empty_document") for f in findings)
                leakage += sum("leak" in f["kind"] or f["kind"]=="workspace_link" for f in findings)
                if findings:
                    errors.extend({"instance":instance,**f} for f in findings)
                for ts in ("shell","files","indexed"):
                    tools = make_toolset(ts, env/"workspace", allow_writes=False)
                    name = "cat" if ts=="shell" else "read_file"
                    for path in ("/workspace/../manifest.json", "/workspace/../evaluation_ref.json", "/manifest.json"):
                        if not tools.call(name,{"path":path}).startswith("Error:"):
                            raise ValueError("tool exposed evaluator path")
                    if FORBIDDEN.search(json.dumps(tools.specs())):
                        raise ValueError("tool schema contains private labels")
                    if ts=="indexed":
                        if set(tools._build_index()["docs"])!={f["path"] for f in m["files"]}:
                            raise ValueError("index inventory mismatch")
                # Answer and evidence identity invariance across intended clutter levels.
                variants = [materialize_world(m["workspace_domain"],m["generation_seed"],level)[2] for level in LEVELS]
                canonical = lambda tasks: [(t["template"],t["answer"],t["required"],t["rows"]) for t in tasks]
                if any(canonical(v)!=canonical(variants[0]) for v in variants[1:]):
                    raise ValueError("answer or evidence changes across difficulty levels")
                invariant_worlds.add(m["world_id"])
            with tempfile.TemporaryDirectory(prefix="fsbench-oracle-") as temp:
                workspace = env/"workspace"
                if m["task"]["output_path"]:
                    workspace = Path(temp)/"workspace"
                    shutil.copytree(env/"workspace",workspace)
                tools = make_toolset("files",workspace,allow_writes=bool(m["task"]["output_path"]))
                answer = solve(m,tools)
                result = score_realistic(answer,m["ground_truth"],workspace)
                citations = score_citations(answer,m)
                if not result["success"] or citations["evidence_recall"]!=1 or citations["evidence_precision"]!=1:
                    raise ValueError("content-derived oracle does not match semantic answer and evidence")
            solvable += 1
            oracle_ok += 1
        except Exception as e:
            errors.append({"instance":instance,"kind":"validation_error","error":str(e)[:300]})
    overlaps = [w for w,splits in world_splits.items() if len(splits)>1]
    if overlaps:
        errors.append({"kind":"split_overlap","worlds":overlaps})
    gate = {"five_domains":len(domains)>=5,"five_templates_per_domain":all(len(templates[d])>=5 for d in DOMAINS),
            "hundred_worlds":len(world_splits)>=100,"multiple_levels":len(levels)>=2,
            "held_out_splits":set.union(*world_splits.values())>={"dev","validation","test"} if world_splits else False,
            "oracle_and_leakage_checks":not errors}
    report = {"benchmark_version":version,"tasks":len(envs),"worlds":len(world_splits),"solvable":solvable,
        "oracle_success":oracle_ok,"answer_leakage":leakage,"missing_evidence":missing,"parser_errors":parser_errors,
        "answer_invariant_worlds":len(invariant_worlds),"automated_valid":not errors,
        "release_criteria":gate,"human_validation":"not established by automated checks",
        "errors":errors,"coverage":{"domains":sorted(domains),"templates":{k:sorted(v) for k,v in templates.items()},"levels":sorted(levels)}}
    target = Path(out) if out else Path(root)/"validation.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def benchmark_card(root, *, out=None, human_runs=None):
    root = Path(root)
    dataset = json.loads((root/"benchmark.json").read_text(encoding="utf-8"))
    validation = json.loads((root/"validation.json").read_text(encoding="utf-8")) if (root/"validation.json").exists() else None
    status = f"{validation['oracle_success']}/{validation['tasks']} content-derived oracle checks passed" if validation else "not yet run"
    baselines = [json.loads(p.read_text(encoding="utf-8")) for p in Path(human_runs).rglob("human.json")] if human_runs else []
    baselines = [b for b in baselines if b.get("benchmark_version")==dataset["benchmark_version"]]
    human_status = (f"Supplied baseline records: {len(baselines)} sessions, {len({b['participant'] for b in baselines})} pseudonymous participants, "
                    f"{len({b['env_id'] for b in baselines})} task instances. Measurement modes: {dict(Counter(b['measurement_mode'] for b in baselines))}. "
                    "These counts report supplied observations, not a certification of human validity.") if baselines else "No human baseline records were supplied for this card."
    text = f'''# FSBench Realistic benchmark card

Benchmark version: **{dataset['benchmark_version']}**. Software version is recorded independently as `package_version` on every run.
Status: **candidate; human validation and external validity have not been established**.

## Purpose and intended use
Evaluate file navigation, source selection, reconciliation, cross-format reasoning, provenance and workflow execution in fictional workplace shares. Synthetic FSBench remains available for controlled interventions; this pack tests authored workplace scenarios and is intended for hypothesis development about ecological validity.

## Non-goals
This is not a validated predictor of workplace productivity, legal/accounting competence, general intelligence, or safety. It does not prove human equivalence. Human-looking folders and successful automated oracles alone do not establish external validity. No public release or leaderboard is authorized by this card.

## Task families and coverage
{dataset['worlds']} unique generated worlds and {dataset['tasks']} task instances across finance, legal, HR, research, and engineering. Five distinct templates per domain cover retrieval, multi_file_reasoning, source_selection, reconciliation, workflow_execution, provenance, and temporal_reasoning. Sibling tasks share a workspace and stay in the same split. Light/routine/dense are authored conditions, not measured difficulty scores.

## Dataset generation
All contents and identities are fictional. PDF, XLSX, CSV, DOCX, JSON and text evidence is rendered deterministically from private world seeds. Authored filenames, stale copies, signed amendments, logs and draft artifacts supply contextual messiness. Modification times can disagree with authority; synthetic creation times are recorded in File activity.csv, while filesystem creation time is not assumed portable. No personal data was collected.

The task recipe is locked to SHA-256 `{dataset['recipe_sha256']}`. Changing task-generating source requires a new benchmark version. The current pack is a candidate; no human-reviewed version has been released.

## Metrics
Semantic `answer_correct`/success is separate from `evidence_precision` and `evidence_recall`. Citation normalization tolerates path separators, case, Markdown links, line suffixes and unambiguous basenames. Wrong or nonexistent sources reduce attribution scores without overriding a correct semantic answer. Provenance tasks assess the required source set, not an agent's internal causal reasoning. Report calls, files read, tokens, cost, latency and failure types separately; there is no universal score.

## Human baselines
{human_status}

No real participants or results are implied by this pack. Collect pseudonymous trials with `human-run`; standardized CLI events are instrumented, native-interface file/search histories are explicitly self-reported. Record final-answer time before post-task citation/rating questions. Never compare human tokens. The protocol requires voluntary participation and preserves raw task ambiguity feedback rather than treating ratings as task scores.

## Validation
Automated validation: {status}. Checks cover actual bytes/timestamps, parsers, required evidence, evaluator-path isolation, structural answer-label leakage, index inventory, answer invariance and an independent content-derived oracle. Leakage scans detect reserved labels/metadata, not every possible semantic shortcut. An oracle supplies source paths and therefore checks solvability rather than discovery difficulty.

## Held-out access and overfitting
Public dev tasks contain manifests. Validation/test tasks contain only an opaque evaluator reference plus the visible task/workspace; manifests and generation seeds live under evaluator-only `.private` storage. `FSBENCH_EVALUATOR_ROOT` can point to separately permissioned storage. Default developer inspection/oracle commands reject held-out labels without `--evaluator`. This is workflow isolation, not protection from a machine owner who can read all files. For a public held-out service, use separate accounts/hosts and keep test workspaces undisclosed until evaluation. Do not publish raw evaluator run directories: transcripts can contain answers.

## Known limitations
Authored, repetitive templates and bounded domains; fictional language and relatively small workspaces; no human expert review yet; possible shortcut exploitation; provider stochasticity even at temperature zero; dependence on interface and extraction support; self-reported native file/search histories; no demonstrated correlation with real deployment outcomes. The activity export provides a global filename inventory and may make discovery easier than a laptop without that artifact. Tasks requiring judgment beyond the documented rules are out of scope. Human studies and correlations must account for sibling tasks and participants rather than treating every task as independent.

## Reproducibility
From the repository, install `.[dev,plot]` and run:

```powershell
fsbench build-benchmark --task-pack realistic-v1 --num-worlds {dataset['worlds']} --seed-file PRIVATE_SEEDS.json --out datasets/realistic-v1
fsbench validate-benchmark --benchmark realistic-v1 --root datasets/realistic-v1
fsbench benchmark-card --root datasets/realistic-v1 --out BENCHMARK_CARD.md
fsbench human-run --env datasets/realistic-v1/dev/INSTANCE_ID --participant anonymous_01
fsbench calibrate --runs runs/realistic-study --human-runs runs/human --out runs/calibration
```

The private seed list used for this local dataset is retained in its `.private/seeds.json`; regenerating it requires evaluator access. Public code does not embed held-out seeds. `build-testset --task-pack realistic-v1 --num-worlds 100 --seed-file PRIVATE_SEEDS.json` creates a separate test-only set. Do not reuse seeds from a public dev dataset.

## Validation study status
Task generation does not establish study completion. The planned first study uses 20 worlds, three task families and three models; attach its actual outcome report when collected. Model IDs and participant availability must be chosen explicitly. Do not substitute oracle or scripted runs for human participants or model evidence.
'''
    target = Path(out) if out else root/"BENCHMARK_CARD.md"
    target.write_text(text,encoding="utf-8")
    return target
