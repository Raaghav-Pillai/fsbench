# Toolset x filename noise experiment

See the [completed 240-run results](toolset-interaction-results.md) for measured outcomes and limitations.

This study crosses `shell`, `files`, and `indexed` with filename noise 0, 0.3, 0.6, and 0.9 on seeds 1–20: 240 trials on the same 80 generated environments. It tests whether the same model's sensitivity to filename degradation changes with retrieval capabilities. No new filesystem variable or combined score is introduced.

## Run

Use Python 3.10+, install `python -m pip install -e ".[dev,plot]"`, and configure `OPENROUTER_API_KEY`. These commands reuse the [filename experiment](filename-noise.md) environments without regenerating them:

```powershell
fsbench validate --envs envs/exp_noise_v2 --filename-noise --index
fsbench run --envs envs/exp_noise_v2 --model openai/gpt-4o-mini --toolset shell files indexed --out runs/exp_noise_toolsets_v1 --experiment filename_noise_toolset_interaction_v1 --order-seed 7 --temperature 0 --max-steps 40 --skip-existing
fsbench evaluate --runs runs/exp_noise_toolsets_v1 --out runs/exp_noise_toolsets_v1/results.csv --paired --interaction --bootstrap-seed 7
fsbench plot --runs runs/exp_noise_toolsets_v1 --paired --x filename_noise --out runs/exp_noise_toolsets_v1/plots
fsbench compare-trace --runs runs/exp_noise_toolsets_v1 --seed 7 --noise 0.9
```

The prior study's model trials are not reused: all three toolsets get fresh trials in the new interleaved schedule. Each seed's 12 noise/toolset combinations are shuffled with a stream keyed by the experiment-order seed and the three environment seeds. Seed blocks are processed in sorted order. This spreads interface and noise conditions across time without assuming an outcome.

`experiment.json` records the full schedule, model settings, expected cells and input fingerprints before the first trial. Execution order is one-based and survives resume. A changed plan or a completed run with incompatible experiment, order, input or model settings is rejected; use a new output directory for a new design. `--skip-existing` retains completed failures as observations as well as successes. An interrupted trial without final metrics is retried in its scheduled slot.

## Input and tool fairness

All toolsets reference the same source environment. Each trial gets the runner's existing private workspace for scratch writes; this is not a separate environment generation. The workspace fingerprint is verified before the model starts. The source workspace stays unchanged, and trial writes cannot leak into another trial.

Preflight validates file inventory and bytes, and builds each index separately from timing the model trial. Trial indexes are fresh and lazy; measured tool time includes their construction. Manifests, task answers, document roles and relevance labels remain outside the agent-visible root. Index extraction errors are reported instead of silently producing empty indexed documents.

`files_read` counts explicit agent reads, not the backend scans performed by search and indexing. Those scans contribute to tool time. Compare both when assessing retrieval cost.

- `shell`: simulated filesystem commands including `find`, `grep`, `cat`, and `convert_to_text`. Raw `grep` does not extract binary document text, and `cat` refuses binary files. This is the existing shell interface, not arbitrary terminal access.
- `files`: format-aware substring search, listing, globbing and reads.
- `indexed`: all `files` tools plus deterministic BM25 keyword ranking over visible paths and extracted contents. It is not semantic embedding search and still includes filename terms. The agent may choose not to use the index.

Tool descriptions, model system prompt, temperature, turn limit, output limit and write capability stay fixed except for the available tool schemas. Different schemas naturally change input-token costs; those differences are part of the interface comparison.

## Pairing and outputs

`evaluate --paired` checks the complete noise x toolset matrix for each world/task/layout seed triple. It warns about missing or duplicate cells and rejects different initial input fingerprints at the same noise. A seed with any missing required cell is excluded from every toolset and noise condition in the paired export. Legacy experiments without factorial metadata retain their original pairing behavior.

`--interaction` requires one model/task/experiment/fixed configuration. It writes:

- `summary.csv`: correctness, efficiency and failure rates by toolset and noise.
- `within_toolset.csv`: paired noise 0.9 minus noise 0 differences.
- `sensitivity.csv`: the compact calls/reads/tokens/success/cost/time interaction summary.
- `between_toolsets.csv`: paired toolset differences at every noise value.
- `interaction.csv`: paired differences of noise sensitivities between toolsets.
- `failures.csv`: counts by toolset, noise and existing failure type, including successful runs.
- `interaction.json` and `interaction.md`: bootstrap settings, data and a readable overview.

Contrasts cover calls, reads, tokens, API cost, wall time and success, using 10,000 paired bootstrap resamples with a stable seed. Every interval explicitly reports whether it includes zero. Null metric pairs are excluded with the pair count reported. These are exploratory, unadjusted intervals, not automatic significance decisions. All-correct trials cannot establish equal population accuracy.

Plots contain one line per model/toolset and use the existing mean ± standard-error style. They include calls, reads, tokens, success, read precision and first-evidence steps, plus cost, latency and candidate exposure. Plot error bars are not the paired bootstrap intervals.

`compare-trace` annotates actual reads of required evidence, stale/draft traps and distractors, as well as invalid paths and tool errors. It rejects ambiguous or mismatched trials rather than comparing different tasks or input files. Labels come from hidden metadata during post-run diagnostics only; agents never receive them.

## Interpretation

Compare workload, correctness, and failure mechanisms separately. A smaller noise sensitivity may mean better retrieval, but could also reflect a high baseline cost, an agent that does not use the added capability, or informative directories/content search. Do not call one interface best from one metric. With 20 seeds and one model trial per cell, uncertainty from model sampling and provider routing remains.
