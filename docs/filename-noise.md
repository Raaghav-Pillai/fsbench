# Filename noise experiment

See the [initial 80-run results](filename-noise-results.md) for measured outcomes and limitations.

`filename_noise` is a controlled intervention on names, with values from 0 through 1. Hold the world, task, and layout seeds and every other configuration value fixed when comparing noise levels. A sweep does this automatically.

Each document gets a stable threshold derived from its world seed and document ID. It is renamed when that threshold is below the requested noise. Copies share the selection threshold, with separate opaque filename suffixes. Thus the set renamed at 0.3 is contained in the sets renamed at 0.6 and 0.9. Small environments need not have exactly the requested fraction renamed.

Selected filenames progressively lose information:

| Noise | Selected filenames retain |
| --- | --- |
| 0 | Original descriptive names; none selected |
| (0, 1/3] | Lexical context, with status, version, date and numeric identifier tokens removed |
| (1/3, 2/3] | Document type plus an opaque ID |
| (2/3, 1] | `doc` plus an opaque ID |

The parameter accepts any real value in [0, 1]. Individual files change at deterministic thresholds and stage boundaries. Increasing noise changes both prevalence and severity; this experiment measures their combined effect, not either independently. The suffix is a hash of world seed and document identity/copy, never the answer or evidence role. Extensions remain unchanged. Clean paths are reserved and collisions are resolved in stable identity order.

Directory assignment and folder splitting are independent of filenames. Selection of distractors, stale versions, duplicates and formats stays fixed. PDF and Office metadata are deterministic, so matching document copies retain identical bytes across conditions. Manifests record `doc_id`, path, clean name, selection threshold and SHA-256; required evidence continues to use stable IDs. `filename_noise_version=2` distinguishes this intervention from the earlier random-name implementation. Regenerate environments before comparing results with this version.

## Reproduce the 80-run experiment

Install with `python -m pip install -e ".[dev,plot]"` and set `OPENROUTER_API_KEY` in the environment or `.env`. Model runs use the configured OpenRouter account.

```powershell
fsbench sweep --task conflict --out envs/exp_noise_v2 --set distractors=50 --set mime_diversity=7 --vary filename_noise=0,0.3,0.6,0.9 --replicates 20 --seed-offset 1
python scripts/verify_filename_noise.py envs/exp_noise_v2
fsbench run --envs envs/exp_noise_v2 --model openai/gpt-4o-mini --toolset files --out runs/exp_noise_v2 --skip-existing
fsbench evaluate --runs runs/exp_noise_v2 --paired --group-by filename_noise --out runs/exp_noise_v2/results.csv
fsbench plot --runs runs/exp_noise_v2 --paired --x filename_noise --out runs/exp_noise_v2/plots
```

This uses seeds 1–20, natural semantic directories, no extra stale versions or duplicates, 50 distractors with default near-miss fraction 0.5, and all seven allowed formats. The conflict task's inherent draft and superseded-document traps remain present. Seven allowed formats does not guarantee all seven appear in every environment. The model uses the existing defaults: temperature 0, 40 model turns and 4,096 output tokens per response.

New sweeps save the full requested condition list. Paired evaluation warns about missing seeds, including conditions with no completed runs, and filters both the summary and exported CSV. Pairing requires matching world, task and layout seeds and fixed non-swept configuration. For older or manually generated environments, supply `--expect filename_noise=0,0.3,0.6,0.9` to `evaluate` or `plot`.

## Diagnostics and interpretation

- `candidate_files_seen`: distinct manifest file paths returned by tools strictly before the first read of required evidence. Repeated listings count once, directories and truncated-away paths do not count. If evidence is never read, counts all candidates returned during the run. This measures exposure, not the model's internal attention.
- `steps_to_first_required_evidence`: one-based tool-call index of the first required-document read, including earlier failed calls. It is null if no required evidence is read. The existing `steps_to_first_required` remains an alias. Search snippets do not count as full reads.

Neither changes correctness scoring. Compare success and score alongside calls, reads, precision, tokens, API cost, time, retrieval failures and recovery rate. Plot error bars show standard errors across seeds; they are not confidence intervals for paired differences. Missing first-evidence steps are excluded from its mean, so interpret that plot together with retrieval failure rates.

Accuracy may stay stable while effort grows, decline with effort, or barely change. Directory names and content search still provide information, so little effect is a valid result. The default CLI runs conditions in sorted order; timing and provider drift can affect comparisons. Do not interpret small differences from 20 seeds as proof of an accuracy threshold.
