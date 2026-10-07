# Controlled index use: filename-noise results

**Enforcing index-first retrieval reduced overall navigation effort, but did not remove the filename-noise effect.** All 240 runs answered correctly. Every enforced run used BM25; none of the naturalistic indexed runs did.

## Design and integrity

The study used `openai/gpt-4o-mini` through OpenRouter, the conflict task, temperature 0, a 40-turn limit, and one trial per cell. Three arms—files/naturalistic, indexed/naturalistic, indexed/index_first—ran against filename noise 0, .3, .6, and .9 over 20 paired worlds (seeds 1–20). All other filesystem settings were held fixed, including 50 distractors and MIME diversity 7.

The same 80 `envs/exp_noise_v2` environments were reused. Execution was interleaved within each world with order seed 7. The completed-run audit verified all 240 planned cells, all 80 unchanged input fingerprints, and index use before reads in every index-first trial. All runs were valid, all required evidence was read, and there were no tool errors. Reported API cost was **$0.148404**.

Collection finished on 2026-10-07, with run completions from 06:02:31 to 06:22:38 UTC. This experiment launched after the phase-1 policy implementation and its 124-test/index-integrity validation, before the later runner/statistics refactor and version bump. Its metadata retains the pre-bump package version and original plan. The running process kept the same loaded agent and policy implementation; subsequent evaluation added the new behavioral diagnostics. Development and regression tests overlapped collection, so latency comparisons are exploratory.

## Cell means

Each row represents 20 paired worlds. All rows have 100% observed success.

| Interface / policy | Noise | Calls | Files read | Total tokens | Steps to first evidence |
|---|---:|---:|---:|---:|---:|
| files / naturalistic | 0 | 7.15 | 1.85 | 4,730.55 | 6.30 |
| files / naturalistic | .9 | 8.35 | 3.00 | 4,988.35 | 7.25 |
| indexed / naturalistic | 0 | 3.95 | 1.55 | 3,060.00 | 3.45 |
| indexed / naturalistic | .9 | 5.75 | 3.00 | 3,725.55 | 4.60 |
| indexed / index_first | 0 | 2.15 | 1.15 | 2,551.30 | 2.05 |
| indexed / index_first | .9 | 3.95 | 2.90 | 3,273.35 | 3.55 |

The indexed naturalistic arm used `search_index` in **0/80** runs. The index-first arm used it in **80/80**, exactly once per run. This removes the earlier ambiguity between index availability and actual use.

At noise .9, index-first used 4.40 fewer calls than files/naturalistic (paired 95% bootstrap interval −5.30 to −3.50) and 1,715 fewer tokens (−2,178.20 to −1,206.45). Compared with indexed/naturalistic, it used 1.80 fewer calls (−2.40 to −1.30); the token difference was −452.20 with an interval of −940.10 to 27.20, which includes zero.

## Filename-noise sensitivity

Changes below are noise .9 minus noise 0, using 10,000 paired-world bootstrap resamples with seed 7. Brackets give percentile 95% intervals.

| Interface / policy | Change in calls | Change in files read | Change in total tokens |
|---|---:|---:|---:|
| files / naturalistic | +1.20 [0.70, 1.85] | +1.15 [1.00, 1.30] | +257.80 [33.65, 473.15] |
| indexed / naturalistic | +1.80 [1.15, 2.55] | +1.45 [1.25, 1.65] | +665.55 [288.15, 1,111.85] |
| indexed / index_first | +1.80 [1.55, 2.05] | +1.75 [1.50, 1.95] | +722.05 [492.45, 1,055.45] |

The index-first arm has a lower effort level, but its increase under noise remains. Its call-sensitivity difference relative to indexed/naturalistic is 0.00 [−0.80, 0.70], and its token-sensitivity difference is +56.50 [−489.75, 569.00]; both intervals include zero. Relative to files/naturalistic, its token increase is larger by 464.25 [134.40, 858.00]. These observations do not support claiming that enforced BM25 eliminates sensitivity or that one interface is universally best.

## What the trajectories show

In index-first runs, mean stale/draft reads before the first required document increased from **0.05 to 1.50** as noise rose from 0 to .9. Mean searches before evidence stayed near one: 1.00 versus 1.05. This associates the extra work with reading competing versions after retrieval, rather than repeatedly searching.

For example, seed 7 at noise .9 issued one index query, read a stale/draft document, read another competing version, then read the required agreement at step 4. The new `inspect` output identifies those actions automatically. It is consistent with a source-selection burden after discovery; it does not establish the model's internal reasoning or show that a different ranker would solve the problem. This index also includes filenames in its retrieval text, so it is itself exposed to filename degradation.

Accuracy remains at a ceiling in this study. A single 20/20 cell has a Wilson 95% interval of approximately 83.9%–100%; zero observed failures is not evidence of equal population accuracy. This is still one model and one trial per world/condition. The new composed profiles, multi-model runner, and repeated-trial analysis are available for testing beyond that ceiling. The proposed 1,440-run study has not been launched; it still needs the other model choices.

## Artifacts and reproduction

The local run directory contains:

* [Audit](../runs/exp_noise_policies_v1/audit.json), [saved schedule](../runs/exp_noise_policies_v1/experiment.json), and [per-run results](../runs/exp_noise_policies_v1/results.csv).
* [Paired interaction report](../runs/exp_noise_policies_v1/interaction.md), [statistical output](../runs/exp_noise_policies_v1/statistics.json), and [behavior table](../runs/exp_noise_policies_v1/behavior.csv).
* [Files-read plot](../runs/exp_noise_policies_v1/plots/files_read_vs_filename_noise.png), [token plot](../runs/exp_noise_policies_v1/plots/total_tokens_vs_filename_noise.png), and seven other metric plots.
* [Same-seed trace comparison](../runs/exp_noise_policies_v1/trace_seed7_noise09.txt), full transcripts, raw tool traces, answers, metadata, and metrics.

Generated runs are git-ignored, so these artifact links refer to the local workspace. Rebuild the report without making model calls:

```powershell
python scripts/analyze_tool_policies.py --runs runs/exp_noise_policies_v1
```

See the [v0.2 guide](v0.2.md) for new-run commands, policies, profiles, adapter integration, statistical definitions, and contribution steps. The final implementation passes **140 tests**.
