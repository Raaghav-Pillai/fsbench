"""Voluntary pseudonymous baselines; no fabricated participants or token comparisons."""
import json
import os
import re
import time
from pathlib import Path

from fsbench.benchmark import load_manifest
from fsbench.evaluate import evaluate_run, load_trace
from fsbench.runner import run_agent

REASONS = ("finding_files", "understanding_contents", "determining_latest_version", "combining_information", "unclear_task", "other")


class HumanAgent:
    def __init__(self, participant, *, interface="cli", input_fn=input, output_fn=print):
        self.participant, self.interface = participant, interface
        self.name = "human:"+participant
        self.input, self.output = input_fn, output_fn
        self.answer_time_s = None
        self.reported_files, self.reported_searches = None,None

    def describe(self):
        return {"kind":"human", "interface":self.interface,"protocol_version":"human-v1"}

    def run(self, prompt, tools):
        start = time.perf_counter()
        self.output(prompt)
        if self.interface == "native":
            self.output("Open only this task workspace: "+str(tools.root))
            if os.name == "nt":
                os.startfile(str(tools.root))
            else:
                self.output("Open the displayed directory using your file manager.")
        else:
            self.output("Tools: "+", ".join(tools.available))
            self.output('Enter TOOL_NAME {JSON arguments}; help shows schemas; finish submits your final JSON answer; abort cancels.')
        while self.interface == "cli":
            command = self.input("> ").strip()
            if command == "abort":
                raise KeyboardInterrupt("participant aborted")
            if command == "finish":
                break
            if command == "help":
                self.output(json.dumps(tools.specs(),indent=2))
                continue
            name, _, raw = command.partition(" ")
            try:
                arguments = json.loads(raw or "{}")
            except json.JSONDecodeError:
                self.output("Invalid JSON arguments; try again.")
                continue
            self.output(tools.call(name,arguments))
        while True:
            try:
                answer = json.loads(self.input("Final answer (JSON): "))
                if not isinstance(answer,dict):
                    self.output("Please use the JSON object requested in the task.")
                    continue
                break
            except json.JSONDecodeError:
                self.output("Invalid JSON; formatting can be corrected before submission.")
        self.answer_time_s = time.perf_counter()-start
        evidence = self.input("Supporting file paths (JSON list; blank keeps citations already submitted): ").strip()
        if evidence:
            try:
                answer["evidence"] = json.loads(evidence)
            except json.JSONDecodeError:
                answer["evidence"] = evidence.split(";")
        if self.interface == "native":
            paths = self.input("Files opened, including incorrect ones (JSON path list; blank if unknown): ").strip()
            if paths:
                try:
                    parsed = json.loads(paths)
                    if isinstance(parsed,list):
                        self.reported_files = parsed
                except json.JSONDecodeError:
                    self.output("File history was not a JSON list; recorded as unknown.")
            searches = self.input("Number of searches (blank if unknown): ").strip()
            if searches.isdigit():
                self.reported_searches = int(searches)
        return answer


def human_run(env, participant, out=None, *, interface="cli", toolset="files", input_fn=input, output_fn=print):
    if not re.fullmatch(r"anonymous_[a-zA-Z0-9_-]{1,32}",participant):
        raise ValueError("use a pseudonymous ID such as anonymous_01, not a name or email")
    manifest = load_manifest(env)
    stamp = time.strftime("%Y%m%dT%H%M%S",time.gmtime())
    run = Path(out) if out else Path("runs/human")/participant/(manifest["env_id"]+"_"+stamp)
    if run.exists():
        raise ValueError("human run directory exists; use a new output path")
    consent = input_fn("Voluntary benchmark session: task actions, answers, timing and optional ratings will be saved under your pseudonym. Type yes to proceed: ")
    if consent.strip().lower()!="yes":
        return None
    agent = HumanAgent(participant,interface=interface,input_fn=input_fn,output_fn=output_fn)
    try:
        run_agent(env,agent,run,toolset=toolset)
    except (KeyboardInterrupt,EOFError):
        # Incomplete sessions never appear as completed baselines.
        (run/"aborted.json").write_text(json.dumps({"participant":participant,"status":"aborted"}),encoding="utf-8")
        return None
    meta_path = run/"meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if agent.answer_time_s is None:
        (run/"aborted.json").write_text(json.dumps({"participant":participant,"status":"aborted"}),encoding="utf-8")
        return None
    meta.update(participant=participant,measurement_mode="instrumented_cli" if interface=="cli" else "self_report_native",
                session_wall_time_s=meta["wall_time_s"], wall_time_s=round(agent.answer_time_s,3),
                self_reported_files=agent.reported_files,self_reported_searches=agent.reported_searches)
    meta_path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    metrics = evaluate_run(run)
    try:
        rating = input_fn("Optional difficulty rating, 1 trivial to 5 very difficult (blank skips): ").strip()
        reasons = input_fn("Optional causes: "+", ".join(f"{i+1}={r}" for i,r in enumerate(REASONS))+" (comma-separated numbers; blank skips): ").strip()
    except (EOFError,KeyboardInterrupt):
        rating,reasons = "",""
    selected = [REASONS[int(v)-1] for v in reasons.split(",") if v.strip().isdigit() and 1<=int(v)<=len(REASONS)]
    answer = json.loads((run/"answer.json").read_text(encoding="utf-8"))
    events = load_trace(run/"trace.jsonl")
    opened = list(dict.fromkeys(p for e in events for p in e["read"])) if interface=="cli" else agent.reported_files
    from fsbench.citations import normalize_path
    known = {normalize_path(f["path"]):f for f in manifest["files"]}
    incorrect = sum(not (known.get(normalize_path(p),{}).get("doc_id") in manifest["required_doc_ids"]) for p in opened) if opened is not None else None
    baseline = {"participant":participant,"task_id":manifest["task"]["task_id"],"env_id":manifest["env_id"],
        "world_id":manifest.get("world_id",str(manifest["seeds"]["world"])), "task_template":manifest.get("task_template",manifest["task"]["type"]),
        "benchmark_family":manifest.get("benchmark_family","synthetic"),"benchmark_version":manifest.get("benchmark_version","synthetic-v2"),
        "difficulty_level":manifest.get("difficulty_level"),"protocol_version":"human-v1","measurement_mode":meta["measurement_mode"],
        "success":metrics["success"],"wall_time_s":meta["wall_time_s"],"files_opened":len(opened) if opened is not None else None,
        "searches":sum(e["tool"] in ("search_index","search_files","grep","find","glob") for e in events) if interface=="cli" else agent.reported_searches,
        "incorrect_files_inspected":incorrect,"evidence":answer.get("evidence",answer.get("sources",[])),
        "evidence_precision":metrics["evidence_precision"],"evidence_recall":metrics["evidence_recall"],
        "difficulty_rating":int(rating) if rating in ("1","2","3","4","5") else None,"difficulty_causes":selected,
        "voluntary_agreement":True}
    (run/"human.json").write_text(json.dumps(baseline,indent=2),encoding="utf-8")
    (run/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    output_fn(f"Saved baseline to {run}; success={metrics['success']}")
    return baseline
