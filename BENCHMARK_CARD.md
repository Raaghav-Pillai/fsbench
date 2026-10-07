# FSBench Realistic benchmark card

Benchmark version: **realistic-v1**. Software version is recorded independently as `package_version` on every run.
Status: **candidate; human validation and external validity have not been established**.

## Purpose and intended use
Evaluate file navigation, source selection, reconciliation, cross-format reasoning, provenance and workflow execution in fictional workplace shares. Synthetic FSBench remains available for controlled interventions; this pack tests authored workplace scenarios and is intended for hypothesis development about ecological validity.

## Non-goals
This is not a validated predictor of workplace productivity, legal/accounting competence, general intelligence, or safety. It does not prove human equivalence. Human-looking folders and successful automated oracles alone do not establish external validity. No public release or leaderboard is authorized by this card.

## Task families and coverage
100 unique generated worlds and 500 task instances across finance, legal, HR, research, and engineering. Five distinct templates per domain cover retrieval, multi_file_reasoning, source_selection, reconciliation, workflow_execution, provenance, and temporal_reasoning. Sibling tasks share a workspace and stay in the same split. Light/routine/dense are authored conditions, not measured difficulty scores.

## Dataset generation
All contents and identities are fictional. PDF, XLSX, CSV, DOCX, JSON and text evidence is rendered deterministically from private world seeds. Authored filenames, stale copies, signed amendments, logs and draft artifacts supply contextual messiness. Modification times can disagree with authority; synthetic creation times are recorded in File activity.csv, while filesystem creation time is not assumed portable. No personal data was collected.

The task recipe is locked to SHA-256 `fc6c44b3d9b9e3aeafe8a5b06dafa0fe41218fb93dc5d578cd545afeb313d4ac`. Changing task-generating source requires a new benchmark version. The current pack is a candidate; no human-reviewed version has been released.

## Metrics
Semantic `answer_correct`/success is separate from `evidence_precision` and `evidence_recall`. Citation normalization tolerates path separators, case, Markdown links, line suffixes and unambiguous basenames. Wrong or nonexistent sources reduce attribution scores without overriding a correct semantic answer. Provenance tasks assess the required source set, not an agent's internal causal reasoning. Report calls, files read, tokens, cost, latency and failure types separately; there is no universal score.

## Human baselines
No human baseline records were supplied for this card.

No real participants or results are implied by this pack. Collect pseudonymous trials with `human-run`; standardized CLI events are instrumented, native-interface file/search histories are explicitly self-reported. Record final-answer time before post-task citation/rating questions. Never compare human tokens. The protocol requires voluntary participation and preserves raw task ambiguity feedback rather than treating ratings as task scores.

## Validation
Automated validation: 500/500 content-derived oracle checks passed. Checks cover actual bytes/timestamps, parsers, required evidence, evaluator-path isolation, structural answer-label leakage, index inventory, answer invariance and an independent content-derived oracle. Leakage scans detect reserved labels/metadata, not every possible semantic shortcut. An oracle supplies source paths and therefore checks solvability rather than discovery difficulty.

## Held-out access and overfitting
Public dev tasks contain manifests. Validation/test tasks contain only an opaque evaluator reference plus the visible task/workspace; manifests and generation seeds live under evaluator-only `.private` storage. `FSBENCH_EVALUATOR_ROOT` can point to separately permissioned storage. Default developer inspection/oracle commands reject held-out labels without `--evaluator`. This is workflow isolation, not protection from a machine owner who can read all files. For a public held-out service, use separate accounts/hosts and keep test workspaces undisclosed until evaluation. Do not publish raw evaluator run directories: transcripts can contain answers.

## Known limitations
Authored, repetitive templates and bounded domains; fictional language and relatively small workspaces; no human expert review yet; possible shortcut exploitation; provider stochasticity even at temperature zero; dependence on interface and extraction support; self-reported native file/search histories; no demonstrated correlation with real deployment outcomes. The activity export provides a global filename inventory and may make discovery easier than a laptop without that artifact. Tasks requiring judgment beyond the documented rules are out of scope. Human studies and correlations must account for sibling tasks and participants rather than treating every task as independent.

## Reproducibility
From the repository, install `.[dev,plot]` and run:

```powershell
fsbench build-benchmark --task-pack realistic-v1 --num-worlds 100 --seed-file PRIVATE_SEEDS.json --out datasets/realistic-v1
fsbench validate-benchmark --benchmark realistic-v1 --root datasets/realistic-v1
fsbench benchmark-card --root datasets/realistic-v1 --out BENCHMARK_CARD.md
fsbench human-run --env datasets/realistic-v1/dev/INSTANCE_ID --participant anonymous_01
fsbench calibrate --runs runs/realistic-study --human-runs runs/human --out runs/calibration
```

The private seed list used for this local dataset is retained in its `.private/seeds.json`; regenerating it requires evaluator access. Public code does not embed held-out seeds. `build-testset --task-pack realistic-v1 --num-worlds 100 --seed-file PRIVATE_SEEDS.json` creates a separate test-only set. Do not reuse seeds from a public dev dataset.

## Validation study status
Task generation does not establish study completion. The planned first study uses 20 worlds, three task families and three models; attach its actual outcome report when collected. Model IDs and participant availability must be chosen explicitly. Do not substitute oracle or scripted runs for human participants or model evidence.
