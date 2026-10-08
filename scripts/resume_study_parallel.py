"""Resume an immutable FSBench plan with isolated agents and bounded concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading

from fsbench.experiment import validate_environment
from fsbench.models import create_agent
from fsbench.provenance import implementation_fingerprint
from fsbench.runner import run_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    if not 1 <= a.workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    root = Path(a.runs).resolve()
    plan = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if implementation_fingerprint() != plan["implementation_sha256"]:
        raise ValueError("implementation changed since the saved plan")
    schedule = plan["schedule"]
    if any(c["toolset"] != "files" or c["tool_policy"] != "naturalistic" or c["trial_index"] != 1 for c in schedule):
        raise ValueError("this study launcher supports files/naturalistic with one trial")
    for c in {c["env_id"]: c for c in schedule}.values():
        if validate_environment(c["env_dir"]) != {k: c[k] for k in ("workspace_sha256", "input_sha256")}:
            raise ValueError(f"source changed: {c['env_id']}")
    lock, stop = threading.Lock(), threading.Event()
    log = root.parent / "execution_events.jsonl"
    def event(**data):
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with lock:
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
    def run_path(c):
        model = c["model"].replace("/", "__").replace(":", "_")
        return root / model / c["toolset"] / c["env_id"]
    pending = []
    for c in schedule:
        run = run_path(c)
        if (run / "metrics.json").exists():
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            wanted = {k: c[k] for k in ("execution_order", "input_sha256", "workspace_sha256", "trial_index")}
            wanted.update(design_id=plan["design_id"], agent_describe=plan["agent_describe"][c["model"]])
            if any(meta.get(k) != v for k, v in wanted.items()):
                raise ValueError(f"existing result differs from plan: {run}")
            continue
        if run.exists():
            archive = root.parent / "interrupted_attempts" / f"order_{c['execution_order']:03d}"
            # Both source and target are verified inside this study before moving a tree.
            if not run.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root.parent):
                raise ValueError("archive path outside the study")
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise ValueError("interrupted archive already exists")
            shutil.move(str(run), str(archive))
            event(event="archive_interrupted_attempt", execution_order=c["execution_order"],
                  model=c["model"], path=str(archive), cost_usd=None,
                  reason="serial launcher interrupted to switch to bounded concurrency")
        pending.append(c)
    event(event="resume", workers=a.workers, completed=len(schedule)-len(pending), pending=len(pending),
          design_id=plan["design_id"], planned_order="unchanged; completion and actual timing may vary")
    def execute(c):
        if stop.is_set():
            return None
        agent = create_agent(plan["provider"], c["model"], max_steps=plan["max_steps"], temperature=plan["temperature"])
        if agent.describe() != plan["agent_describe"][c["model"]]:
            raise ValueError("agent settings changed")
        context = {**c, "experiment": plan["experiment"], "expected_cells": plan["expected_cells"],
                   "design_version": plan["design_version"], "design_id": plan["design_id"],
                   "provider": plan["provider"], "implementation_sha256": plan["implementation_sha256"],
                   "execution_workers": a.workers, "execution_started_at": datetime.now(timezone.utc).isoformat()}
        event(event="start", execution_order=c["execution_order"], model=c["model"], env_id=c["env_id"])
        result = run_agent(c["env_dir"], agent, run_path(c), toolset=c["toolset"],
                           tool_policy=c["tool_policy"], allow_writes=plan["allow_writes"], run_context=context)
        meta = json.loads((run_path(c) / "meta.json").read_text(encoding="utf-8"))
        err = meta.get("agent_error") or ""
        if any(s in err for s in ("HTTP 401", "HTTP 402", "HTTP 403")):
            stop.set()
        event(event="complete", execution_order=c["execution_order"], model=c["model"],
              success=result["success"], cost_usd=result.get("cost_usd"), agent_error=err)
        return c, result
    done = len(schedule) - len(pending)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(execute, c) for c in pending]
        for future in as_completed(futures):
            try:
                value = future.result()
            except BaseException:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
            if value is None:
                continue
            c, result = value
            done += 1
            print(f"[{done}/{len(schedule)}] order={c['execution_order']} {c['model']} "
                  f"success={result['success']} calls={result['n_calls']} cost={result.get('cost_usd')}", flush=True)
    event(event="finished", completed=done, expected=len(schedule), stopped=stop.is_set())
    if done != len(schedule):
        raise SystemExit("Study stopped after a fatal account/provider error; remaining cells were not executed.")


if __name__ == "__main__":
    main()
