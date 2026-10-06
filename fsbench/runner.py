"""Run an agent against an environment and record everything the evaluator needs.

Run directory layout::

    <run_dir>/
        workspace/     private copy of the env workspace (the agent may write here)
        trace.jsonl    one event per tool call
        answer.json    the agent's final answer
        meta.json      env, agent, toolset, timing, optional usage/cost
        metrics.json   evaluator output
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Protocol

from fsbench.evaluate import evaluate_run, parse_answer
from fsbench.paths import long_path
from fsbench.provenance import snapshot
from fsbench.tools import VROOT, ToolSet, Tracer, make_toolset


class Agent(Protocol):
    name: str

    def run(self, prompt: str, tools: ToolSet) -> dict | str:
        """Solve the task using ``tools.call(name, args)``; return the final answer.

        ``tools.specs()`` gives JSON schemas for function calling. An agent may expose a
        ``usage`` dict (input_tokens, output_tokens, cost_usd, ...) which is saved to meta.json.
        """
        ...


def run_agent(
    env_dir: str | Path,
    agent: Agent,
    run_dir: str | Path,
    *,
    toolset: str = "files",
    allow_writes: bool = True,
    max_output_chars: int = 20000,
) -> dict:
    env, run = long_path(env_dir), long_path(run_dir)
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    shutil.copytree(env / "workspace", run / "workspace")
    task = json.loads((env / "task.json").read_text(encoding="utf-8"))
    manifest = json.loads((env / "manifest.json").read_text(encoding="utf-8"))

    tracer = Tracer(run / "trace.jsonl")
    tools = make_toolset(toolset, run / "workspace", tracer,
                         allow_writes=allow_writes, max_output_chars=max_output_chars)
    error = None
    start = time.perf_counter()
    try:
        raw = agent.run(task["prompt"], tools)
    except Exception:
        raw, error = {}, traceback.format_exc()
    finally:
        tracer.close()
    wall = time.perf_counter() - start

    (run / "answer.json").write_text(json.dumps(parse_answer(raw), indent=2), encoding="utf-8")
    transcript = getattr(agent, "transcript", None)
    if transcript:
        (run / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    describe = getattr(agent, "describe", None)
    agent_describe = describe() if callable(describe) else None
    sweep = manifest.get("sweep") or {}
    (run / "meta.json").write_text(json.dumps({
        "env_dir": str(Path(env_dir).resolve()),
        "env_id": manifest.get("env_id"),
        "agent": agent.name,
        "agent_describe": agent_describe,
        "model": (agent_describe or {}).get("model"),
        "task": manifest["task"]["type"],
        "toolset": toolset,
        "allow_writes": allow_writes,
        "max_output_chars": max_output_chars,
        "world_seed": manifest["seeds"]["world"],
        "filename_noise": manifest["config"]["filename_noise"],
        "filename_noise_version": manifest.get("filename_noise_version", 1),
        "condition_label": (manifest.get("condition") or {}).get("label"),
        "seeds": manifest["seeds"],
        "config": manifest["config"],
        "condition": manifest.get("condition"),
        "sweep_replicate": sweep.get("replicate"),
        **snapshot(),
        "wall_time_s": round(wall, 3),
        "agent_error": error,
        "raw_answer": raw if isinstance(raw, str) else None,
        "usage": getattr(agent, "usage", None),
    }, indent=2), encoding="utf-8")
    metrics = evaluate_run(run_dir)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


class OracleAgent:
    """Reads exactly one copy of each required doc and submits the ground truth.

    Validates that an environment is solvable through a given toolset and that the
    evaluator scores a perfect run as success with known_path_regret 1.0.
    """

    name = "oracle"
    READ_TOOL = {"shell": "convert_to_text", "files": "read_file", "indexed": "read_file"}

    def __init__(self, manifest: dict):
        self.manifest = manifest

    def describe(self) -> dict:
        return {"kind": "oracle"}

    def run(self, prompt: str, tools: ToolSet) -> dict:
        m = self.manifest
        for doc_id in m["required_doc_ids"]:
            copies = [f for f in m["files"] if f["doc_id"] == doc_id]
            best = min(copies, key=lambda f: (f["path"].count("/"), f["path"]))
            tools.call(self.READ_TOOL[tools.name], {"path": f"{VROOT}/{best['path']}"})
        gt = m["ground_truth"]
        out = m["task"]["output_path"]
        if out:
            lines = ["invoice_id,amount_due", *(f"{r['invoice_id']},{r['amount_due']}" for r in gt["rows"])]
            tools.call("write_file", {"path": f"{VROOT}/{out}", "content": "\n".join(lines) + "\n"})
            return {"output_path": f"{VROOT}/{out}"}
        return {k: gt[k] for k in m["task"]["answer_schema"]}
