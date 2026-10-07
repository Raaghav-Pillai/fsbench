# Filename noise: initial 80-run experiment

GPT-4o-mini answered all 80 tasks correctly. Higher noise was associated with more reads and higher average token use in this sample; no accuracy degradation was observed through noise 0.9. The paired intervals are wide, so this is preliminary evidence about efficiency, not a precise effect estimate or an accuracy threshold.

## Setup and validation

Used `openai/gpt-4o-mini` through OpenRouter, the `conflict` task, `files` tools, 50 distractors, seven allowed document formats, and 20 paired world/task/layout seeds (1–20). Noise levels were 0, 0.3, 0.6 and 0.9. Other filesystem variables retained their defaults. Temperature was 0, with a 40-turn limit.

All 80 runs survived paired filtering. Before model execution, the integrity check verified all 4,240 files: matching document identities, directory assignment, file bytes, answers and required evidence, with monotonically nested renamed sets. The final regression suite passed all 104 tests.

The [experiment guide](filename-noise.md) gives the exact commands and explains intervention version 2. Generated environments are under `envs/exp_noise_v2`; local results, traces, CSV, plots, an analysis script and paired comparisons are under `runs/exp_noise_v2`. These generated artifacts are ignored by Git.

## Results

Effort columns are averages per run. API costs are reported by OpenRouter.

| Noise | Correct | Tool calls | Files read | Tokens | Cost/run (USD) | Wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 20/20 | 7.10 | 1.85 | 4,551 | 0.000750 | 7.30 |
| 0.3 | 20/20 | 7.40 | 2.10 | 4,743 | 0.000816 | 7.66 |
| 0.6 | 20/20 | 8.10 | 2.60 | 4,954 | 0.000852 | 7.79 |
| 0.9 | 20/20 | 8.25 | 3.50 | 5,610 | 0.000917 | 7.40 |

Total reported API cost: **$0.066715**. No run failures occurred. All runs read the required evidence.

From noise 0 to 0.9, mean calls increased 16%, reads 89%, and tokens 23%. Mean read precision fell from 0.575 to 0.321. First required evidence moved from tool-call step 6.25 to 7.10. Candidate exposure did not increase consistently (3.40, 4.05, 4.05, 3.50), and wall time showed no consistent increase.

The paired mean difference at noise 0.9 was **+1.15 calls** (95% bootstrap interval: -0.30 to +2.85) and **+1,059 tokens** (-312 to +3,263). These percentile intervals use 10,000 resamples of the 20 paired seed differences, with bootstrap seed 7. Both include zero. A high-noise trial requiring 22 calls contributed substantial variation.

## Interpretation

The observed pattern is consistent with filename information helping efficiency while content search and informative directories preserve accuracy. For example, one paired seed selected the signed contract after two searches at noise 0.3; at noise 0.9 it read all three candidate contracts after the same two searches. This example illustrates a mechanism, not a universal strategy.

Only 20 seeds were sampled, with one model trial per condition and seed. Even 20/20 correct has a Wilson 95% interval of approximately 84%–100%. The experiment cannot establish that accuracy never degrades. It also changes both renaming prevalence and semantic severity, so their separate effects cannot be identified here.

Conditions ran in sorted order (0.3, 0.6, 0.9, then 0), making provider drift a possible confounder, especially for latency. Bootstrap intervals describe this sample's paired variation and do not account for that drift or multiple comparisons. Additional seeds and repeated, interleaved model trials would strengthen the estimate.
