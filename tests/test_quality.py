import json

import pytest

from fsbench import FSConfig
from fsbench.evaluate import (
    balance_warnings,
    keep_paired,
    mixed_version_warnings,
    select_envs,
)
from fsbench.paths import long_path
from fsbench.runner import OracleAgent, run_agent
from fsbench.sweep import run_sweep
from fsbench.tasks import TASK_TYPES, _money_eq, _norm
from fsbench.tools import FileTools, IndexedTools, Tracer, VROOT
from fsbench.world import build_world

from test_evaluate import ScriptedAgent


def test_meta_has_full_provenance(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(), label="organized")
    run_agent(env, OracleAgent(m), tmp_path / "run")
    meta = json.loads((long_path(tmp_path / "run" / "meta.json")).read_text(encoding="utf-8"))
    assert meta["fsbench_version"]
    assert meta["timestamp"].endswith("Z")
    assert meta["python_version"]
    assert meta["seeds"] == m["seeds"]
    assert meta["config"] == m["config"]
    assert meta["condition"]["label"] == "organized"
    assert meta["condition"]["config_hash"] == m["condition"]["config_hash"]
    assert meta["task"] == "conflict"
    assert meta["world_seed"] == m["seeds"]["world"]
    assert meta["agent_describe"]["kind"] == "oracle"


def test_select_envs_is_balanced(tmp_path):
    run_sweep(["conflict"], FSConfig(distractors=5), {"depth": [0, 3]}, tmp_path / "envs",
              replicates=4, seed_offset=1)
    from fsbench.evaluate import find_envs
    envs = find_envs(tmp_path / "envs")
    picked = select_envs(envs, replicates=2, only="depth")
    labels = []
    for env in picked:
        m = json.loads((long_path(env) / "manifest.json").read_text(encoding="utf-8"))
        labels.append((m["condition"]["label"], m["sweep"]["replicate"], m["seeds"]["world"]))
    assert sorted(labels) == [
        ("depth=0", 0, 1), ("depth=0", 1, 2),
        ("depth=3", 0, 1), ("depth=3", 1, 2),
    ]
    assert select_envs(envs, only="filename_noise") == []


def test_balance_warning_and_paired_filter():
    rows = [
        {"agent": "a", "toolset": "files", "task_type": "conflict", "sweep.vars": "depth=0",
         "seed.world": 1, "fsbench_version": "0.1.0", "model": "m"},
        {"agent": "a", "toolset": "files", "task_type": "conflict", "sweep.vars": "depth=0",
         "seed.world": 2, "fsbench_version": "0.1.0", "model": "m"},
        {"agent": "a", "toolset": "files", "task_type": "conflict", "sweep.vars": "depth=6",
         "seed.world": 1, "fsbench_version": "0.1.0", "model": "m"},
    ]
    warns = balance_warnings(rows)
    assert warns and "missing" in warns[0] and "2" in warns[0]
    paired = keep_paired(rows)
    assert {r["seed.world"] for r in paired} == {1}
    mixed = mixed_version_warnings(rows + [{**rows[0], "fsbench_version": "0.2.0"}])
    assert any("fsbench_version" in w for w in mixed)


@pytest.mark.parametrize("task", sorted(TASK_TYPES))
def test_decoys_differ_from_ground_truth(task):
    for seed in range(12):
        inst = TASK_TYPES[task].build(build_world(seed), seed)
        gt = inst.ground_truth
        assert inst.decoy_answers
        for doc_id, decoy in inst.decoy_answers.items():
            if task == "workflow":
                assert decoy["rows"] != gt["rows"], (seed, doc_id)
            elif task == "reconcile":
                same_bal = _money_eq(decoy.get("outstanding_balance"), gt["outstanding_balance"])
                same_ids = decoy.get("unpaid_invoices") == gt["unpaid_invoices"]
                assert not (same_bal and same_ids), (seed, doc_id)
            elif task == "conflict":
                same_val = _money_eq(decoy.get("annual_value"), gt["annual_value"])
                same_date = decoy.get("renewal_date") == gt["renewal_date"]
                same_id = _norm(decoy.get("contract_id")) == _norm(gt["contract_id"])
                assert not (same_val and same_date and same_id), (seed, doc_id)
            else:
                assert any(_norm(decoy.get(k, "")) != _norm(gt[k]) for k in ("employee_id", "job_title", "manager")), (
                    seed, doc_id)


def _read_required(m):
    return [("read_file", {"path": f"{VROOT}/{f['path']}"})
            for f in m["files"] if f["role"] == "required" and f["copy"] == 0]


def test_failure_retrieval(make_env, tmp_path):
    env, m = make_env("conflict")
    metrics = run_agent(env, ScriptedAgent([], {
        "contract_id": "MSA-000-000", "renewal_date": "2020-01-01", "annual_value": 1,
    }), tmp_path / "run")
    assert metrics["failure_type"] == "retrieval_failure"
    assert not metrics["evidence_found"] and not metrics["success"]


def test_failure_stale_version(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(stale_version_rate=1.0))
    trap_id = next(d for d in m["trap_doc_ids"] if d in m["decoy_answers"])
    decoy = m["decoy_answers"][trap_id]
    trap = next(f for f in m["files"] if f["doc_id"] == trap_id)
    calls = _read_required(m) + [("read_file", {"path": f"{VROOT}/{trap['path']}"})]
    metrics = run_agent(env, ScriptedAgent(calls, decoy), tmp_path / "run")
    assert metrics["evidence_found"]
    assert metrics["failure_type"] == "stale_version_confusion"
    assert metrics["failure_detail"] == trap_id


def test_failure_reasoning(make_env, tmp_path):
    env, m = make_env("conflict")
    metrics = run_agent(env, ScriptedAgent(_read_required(m), {
        "contract_id": "MSA-000-000", "renewal_date": "2099-12-31", "annual_value": 0.01,
    }), tmp_path / "run")
    assert metrics["evidence_found"]
    assert metrics["failure_type"] == "reasoning_failure"


def test_failure_format(make_env, tmp_path):
    env, m = make_env("retrieve")
    metrics = run_agent(env, ScriptedAgent([], "not json at all"), tmp_path / "run")
    assert metrics["failure_type"] == "output_format_failure"


def test_failure_write(make_env, tmp_path):
    env, m = make_env("workflow")
    metrics = run_agent(env, ScriptedAgent([], {"output_path": m["ground_truth"]["output_path"]}), tmp_path / "run")
    assert metrics["failure_type"] == "write_failure"


def test_failure_timeout(make_env, tmp_path):
    env, m = make_env("retrieve")
    agent = ScriptedAgent([], {"employee_id": "E-1", "job_title": "x", "manager": "y"})
    agent.usage = {**agent.usage, "hit_step_limit": True}
    metrics = run_agent(env, agent, tmp_path / "run")
    assert metrics["failure_type"] == "timeout"


def test_failure_agent_error(make_env, tmp_path):
    class Crashy:
        name = "crashy"

        def run(self, prompt, tools):
            raise RuntimeError("boom")

    env, _ = make_env("retrieve")
    metrics = run_agent(env, Crashy(), tmp_path / "run")
    assert metrics["failure_type"] == "agent_error"


def test_parse_ms_and_index_build_ms(make_env):
    env, m = make_env("conflict", FSConfig(mime_types=("pdf",), distractors=8))
    tracer = Tracer()
    tools = FileTools(env / "workspace", tracer)
    req = next(f["path"] for f in m["files"] if f["role"] == "required")
    tools.call("read_file", {"path": f"{VROOT}/{req}"})
    ev = tracer.events[-1]
    assert ev["parse_ms"] > 0
    assert ev["duration_ms"] >= ev["parse_ms"]

    idx_tracer = Tracer()
    idx = IndexedTools(env / "workspace", idx_tracer)
    idx.call("search_index", {"query": "agreement"})
    assert idx_tracer.events[-1]["index_build_ms"] > 0


def test_timing_totals_on_oracle_run(make_env, tmp_path):
    env, m = make_env("retrieve")
    metrics = run_agent(env, OracleAgent(m), tmp_path / "run")
    assert metrics["wall_time_s"] >= 0
    assert metrics["tool_time_s"] >= 0
    assert metrics["read_time_s"] >= 0
    assert metrics["overhead_s"] >= 0
    assert "known_path_optimal_calls" in metrics
    assert "discovery_optimal_calls" in metrics
