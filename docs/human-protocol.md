# Human baseline and validity protocol (human-v1)

The realistic-v1 pack is an authored candidate, not a human-validated benchmark yet. A content oracle establishes computational consistency, not relevance to deployment. Do not label scripted/oracle test fixtures as participants or report test-fixture timings as human results.

## Review before collection

Have at least one competent reviewer per sampled domain inspect the task and its evidence, using evaluator access. Ask whether the request resembles an actual file-based work task; whether the authority rules are clear; whether the specified evidence really supports the answer; whether alternative valid evidence chains exist; and whether jargon, formatting, or unrealistic shortcuts dominate difficulty. Record the benchmark version, task ID, pseudonymous reviewer ID, decision, ambiguity category, and proposed correction in a facilitator-owned review log. Actual names, employer names, screenshots of personal files, and sensitive professional records are unnecessary.

Any accepted task-data correction must produce a new benchmark version. Preserve the original review and dataset fingerprints rather than silently editing held-out workspaces. An alternative sufficient citation set can be specified in an evaluator manifest through `acceptable_evidence_sets`; its task-data revision requires versioning too.

## Initial pilot

Start with 10–20 participant-task sessions, for example three technically competent volunteers each attempting five tasks drawn from distinct worlds across the five domains. Cover source selection, multi-file reasoning, and provenance. This is a pilot for ambiguity and instrumentation, not a population estimate. Avoid giving one participant multiple sibling tasks from the same world: answering one may reveal another answer.

Use a separate dev-world practice task. Record the eligibility rule (for example, regular use of workplace files/spreadsheets) at study level without collecting identities. Obtain voluntary agreement before timing and allow cancellation at any point. Participants may decline optional ratings. Keep any participant-to-person contact mapping outside FSBench if one is operationally necessary; the benchmark only accepts IDs such as `anonymous_01`.

Randomize or counterbalance task order and record the assignment schedule before collection. Keep instruction wording, extraction support, allowed tools, time limits if any, and interface constant within a comparison. Participants should not consult other people, models, external search, or evaluator files during a trial. Facilitators should record deviations and analyze them separately rather than silently discard inconvenient outcomes.

## Standardized interface (recommended)

```powershell
fsbench human-run --env datasets/realistic-v1/validation/INSTANCE_ID `
  --participant anonymous_01 --toolset files --interface cli
```

The participant sees the same task prompt and file contents as agents. Commands have the form `read_file {"path":"/workspace/legal/signed/IMG_3812.pdf"}`; `help` shows schemas, `finish` submits the answer, and `abort` cancels. Tool actions are recorded without access to evaluator metadata. File reads and searches are measured, not recalled afterward. The response must use the task's requested JSON object; evidence formatting is scored tolerantly and independently from the semantic answer.

Timing begins immediately before the task is displayed. It stops after final-answer submission, before the supporting-evidence question and optional difficulty rating. Post-answer citations are retrospective and may differ from an agent's in-answer citations; report that distinction. Retain both the submitted answer and citations, and do not coach after a wrong response.

## Native interface

```powershell
fsbench human-run --env datasets/realistic-v1/validation/INSTANCE_ID `
  --participant anonymous_02 --interface native
```

On Windows the task's isolated workspace opens in the file manager. File/search histories are self-reported afterward and may be unknown. Unknown is stored as null; do not interpret it as zero. Native and CLI data are stratified in calibration. Native applications may rewrite files or expose more metadata than agent tools, so record this interface difference. Use a separate participant account or computer containing only visible task artifacts when evaluator confidentiality matters; a normal local file manager is not an adversarial sandbox.

## Outcomes and feedback

`human.json` stores pseudonym, task/world/version IDs, semantic success, final-answer time, opened-file/search counts, incorrectly inspected files, citation precision/recall, optional difficulty rating (1–5), and difficulty causes. Human token counts are neither collected nor compared. Causes include navigation, contents, version selection, combining information, unclear task, and other. An unclear task is a validity concern, not evidence that an agent is incapable.

Aborted sessions are marked and excluded from completed-run discovery. Retain a study-level completion/withdrawal count; do not relabel withdrawals as successes. Protect raw traces because queries or volunteered answers can still contain unintended personal information. Agree on access, retention and deletion rules before collection.

## Calibration and analysis

```powershell
fsbench calibrate --runs runs/realistic-study --human-runs runs/human --out runs/calibration
```

Outputs include median human completion time, participant/session counts, success rates, median agent calls, and retrieval failure rate. No filesystem-variable sum is used. Descriptive difficulty labels use median human difficulty ratings: at most 2 is low, at least 4 is high, otherwise medium. At least three distinct participant ratings per task are required by default; repeated sessions by one person do not satisfy that threshold. Unobserved tasks remain unclassified. These thresholds are a transparent pilot convention, not a universal scale or combined FSBench score.

For the deployment-validity question, examine whether task/time/error patterns agree across humans and agents, separately by model, interface, domain and task family. Investigate disagreements through traces and review notes. A subsequent confirmatory study should pre-specify associations, uncertainty estimates and exclusions, cluster by participant and world, separate speed/accuracy tradeoffs, and include representative real deployment tasks. Positive association within this authored pack alone is insufficient proof of external validity.
