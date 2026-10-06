# Toolset x filename noise: 240-run results

All 240 trials answered correctly. Filename degradation increased explicit reads in every interface. The indexed interface had lower absolute call and token counts, but **the model never called `search_index` in any of its 80 indexed trials**. This experiment therefore does not establish that BM25 retrieval removes filename sensitivity.

## Design and validation

Used `openai/gpt-4o-mini`, `conflict`, noise 0/0.3/0.6/0.9, toolsets shell/files/indexed, and 20 paired world/task/layout seeds (1–20). The existing 80 `exp_noise_v2` environments were reused, with 50 distractors and seven allowed formats. Temperature was 0 and the limit was 40 model turns. Every trial was fresh; earlier files-only results were not reused.

Each seed's 12 noise/toolset cells were shuffled reproducibly with order seed 7. All 240 completed cells matched the saved schedule and input fingerprints, passed factorial pairing, and left the 80 source environments unchanged. Each trial's private workspace matched its planned source before the model started. The full test suite passed 117 tests, including index isolation, binary extraction, deterministic ordering, input integrity, complete-cell pairing and CLI resume checks.

Total reported API cost: **$0.223603**. The [experiment guide](toolset-interaction.md) contains the exact CLI recipe and metric definitions. Raw traces, CSVs, nine grouped plots, a reproducible analysis script, and audit results are in the Git-ignored directory `runs/exp_noise_toolsets_v1`.

## Absolute workload at noise 0.9

All columns except correctness are means over the same 20 seeds.

| Toolset | Correct | Calls | Explicit reads | Tokens | Cost/run (USD) | Wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| shell | 20/20 | 12.60 | 2.95 | 9,820 | 0.001554 | 10.92 |
| files | 20/20 | 8.35 | 3.00 | 5,100 | 0.000883 | 7.77 |
| indexed | 20/20 | 5.40 | 3.00 | 3,491 | 0.000617 | 5.14 |

At this noise level, the paired indexed-minus-files difference was -2.95 calls (95% bootstrap interval -3.70 to -2.15) and -1,609 tokens (-2,070 to -1,110). Shell-minus-files was +4.25 calls (+2.35 to +6.65) and +4,720 tokens (+3,008 to +7,041). These compare absolute workload, not sensitivity to noise or index efficacy.

## Sensitivity: noise 0.9 minus noise 0

| Toolset | Delta calls [95% interval] | Delta reads [95% interval] | Delta tokens [95% interval] | Delta success |
| --- | --- | --- | --- | --- |
| shell | +1.85 [-1.05, +5.00] | +1.15 [+0.95, +1.35] | +866 [-2,051, +3,979] | 0 |
| files | +1.35 [+0.65, +2.00] | +1.20 [+1.05, +1.40] | +486 [+28, +875] | 0 |
| indexed | +1.50 [+1.30, +1.70] | +1.50 [+1.30, +1.70] | +447 [+379, +516] | 0 |

Intervals use 10,000 resamples of the paired seed differences, with bootstrap seed 7. The shell call/token intervals include zero; the files and indexed call/token intervals do not. All three read intervals exclude zero. Wall-time-change intervals include zero for all three toolsets.

The direct interaction contrasts are more relevant than comparing separate intervals: indexed-minus-files noise sensitivity was +0.15 calls [-0.50, +0.85] and -39 tokens [-444, +403]. All between-toolset sensitivity intervals for calls and tokens included zero, so this sample does not clearly establish different sensitivities for those metrics.

Indexed read sensitivity was +0.30 reads greater than files [+0.10, +0.50]. Its baseline used fewer reads (1.50 versus 1.80), while both interfaces read 3.00 files at high noise. Lower baseline effort did not mean filename sensitivity disappeared. No single toolset is declared best.

## Tool choice and failure mechanisms

The indexed interface exposed all files tools plus BM25 search, but `search_index` use was **0/20 at every noise level**. Its index passed preflight and regression tests; the model chose other tools during the measured trials. Lower workload in this interface cannot be credited to actual BM25 calls. A change in strategy when the tool menu changes is consistent with the traces, but its cause is not isolated by this experiment.

For seed 7 at noise 0.9, required evidence appeared at step 14 for shell, 8 for files, and 4 for indexed. The indexed trial shortened an unsuccessful `search_files` query; the files trial switched to directory navigation. Shell made several unsuccessful exact-name `find` calls before navigating directories. The saved `trace_seed7_noise09.txt` annotates these reads and traps.

There were no run-level retrieval, reasoning, timeout or infrastructure failures. All runs read the required evidence. Shell encountered eight `cat` binary-file errors and recovered using the available interface. Files encountered one nonexistent-directory error and recovered. Indexed had no tool errors. These recovered tool errors are distinct from task failures; `failures.csv` and the metric summaries preserve that distinction.

## Limits

This was one model trial per cell on 20 seeds. Intervals are exploratory and unadjusted for multiple comparisons. Perfect observed accuracy does not prove equal population accuracy or identify an accuracy-degradation threshold. Provider routing/model variation remains possible despite interleaving. Local validation ran during early trials, so wall/tool timing comparisons should also be treated as exploratory.

`files_read` measures explicit agent reads, not all backend scans. The index was not invoked, so these measured indexed trials do not include runtime index-construction cost. A controlled follow-up that ensures the index is actually used would be needed to answer whether BM25 itself removes filename sensitivity; this study measures the available interface and the model's chosen behavior.
