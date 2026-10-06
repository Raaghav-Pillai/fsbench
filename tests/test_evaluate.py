import json

import pytest

from fsbench import PRESETS, FSConfig
from fsbench.evaluate import evaluate_runs, parse_answer, summarize
from fsbench.paths import long_path
from fsbench.runner import OracleAgent, run_agent
from fsbench.tasks import parse_date, parse_money
from fsbench.tools import VROOT


class ScriptedAgent:
    def __init__(self, calls, answer, name="scripted"):
        self.calls, self.answer, self.name = calls, answer, name
        self.usage = {"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.001}

    def run(self, prompt, tools):
        for name, args in self.calls:
            tools.call(name, args)
        return self.answer


@pytest.mark.parametrize("task", ["retrieve", "reconcile", "conflict", "workflow"])
@pytest.mark.parametrize("toolset", ["shell", "files", "indexed"])
def test_oracle_is_perfect(make_env, tmp_path, task, toolset):
    env, m = make_env(task, PRESETS["horrible"])
    metrics = run_agent(env, OracleAgent(m), tmp_path / "run", toolset=toolset)
    assert metrics["success"] and metrics["score"] == 1.0
    assert metrics["known_path_regret"] == 1.0
    assert metrics["failure_type"] is None
    assert metrics["evidence_found"]
    assert metrics["required_recall_read"] == 1.0
    assert metrics["read_precision"] == 1.0
    assert metrics["path_hallucination_rate"] == 0.0


def test_wrong_answer_traps_and_hallucinations_are_measured(make_env, tmp_path):
    env, m = make_env("reconcile", FSConfig(stale_version_rate=1.0, distractors=20, semantic_similarity=1.0))
    stale = next(f for f in m["files"] if f["doc_id"].startswith("ledger-stale"))
    near = next(f for f in m["files"] if f["role"] == "near")
    req = next(f for f in m["files"] if f["role"] == "required")
    calls = [
        ("list_directory", {"path": VROOT}),
        ("read_file", {"path": "/workspace/invoices/acme_april.pdf"}),
        ("read_file", {"path": f"{VROOT}/{stale['path']}"}),
        ("read_file", {"path": f"{VROOT}/{near['path']}"}),
        ("read_file", {"path": f"{VROOT}/{req['path']}"}),
        ("read_file", {"path": f"{VROOT}/{req['path']}"}),
    ]
    answer = {"outstanding_balance": 1.0, "unpaid_invoices": m["ground_truth"]["unpaid_invoices"]}
    metrics = run_agent(env, ScriptedAgent(calls, answer), tmp_path / "run")

    assert not metrics["success"]
    assert metrics["score"] == 0.5
    assert metrics["n_calls"] == 6
    assert metrics["path_hallucination_rate"] == pytest.approx(1 / 6, abs=1e-3)
    assert metrics["hallucinated_paths"] == ["/workspace/invoices/acme_april.pdf"]
    assert metrics["trap_reads"] == 1 and metrics["near_distractor_reads"] == 1
    assert metrics["trap_exposed"] and not metrics["recovered"]
    assert metrics["redundant_reads"] == 1
    assert metrics["files_read"] == 3
    assert metrics["read_precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["steps_to_first_required"] == 5
    assert metrics["known_path_regret"] == pytest.approx(6 / m["oracle"]["known_path_optimal_calls"], abs=1e-3)
    assert metrics["failure_type"] == "retrieval_failure"
    assert metrics["usage"]["cost_usd"] == 0.001


def test_workflow_scores_the_written_file(make_env, tmp_path):
    env, m = make_env("workflow")
    out = f"{VROOT}/{m['ground_truth']['output_path']}"
    rows = m["ground_truth"]["rows"]

    missing = run_agent(env, ScriptedAgent([], {"output_path": out}), tmp_path / "a")
    assert not missing["success"] and not missing["outcome"]["fields"]["file_exists"]

    good = "invoice_id,amount_due\n" + "".join(f"{r['invoice_id']},\"${float(r['amount_due']):,.2f}\"\n" for r in rows)
    ok = run_agent(env, ScriptedAgent([("write_file", {"path": out, "content": good})], {}), tmp_path / "b")
    assert ok["success"]

    partial = "Invoice_ID,Amount_Due\n" + f"{rows[0]['invoice_id']},{rows[0]['amount_due']}\nINV-0000,5.00\n"
    p = run_agent(env, ScriptedAgent([("write_file", {"path": out, "content": partial})], {}), tmp_path / "c")
    assert not p["success"] and 0 < p["score"] < 1

    written = long_path(tmp_path / "b" / "workspace").joinpath(*m["ground_truth"]["output_path"].split("/"))
    assert written.is_file()
    assert not long_path(env / "workspace").joinpath(*m["ground_truth"]["output_path"].split("/")).exists()


def test_agent_crash_is_recorded_not_raised(make_env, tmp_path):
    class Crashy:
        name = "crashy"

        def run(self, prompt, tools):
            tools.call("list_directory", {})
            raise RuntimeError("boom")

    env, _ = make_env("retrieve")
    metrics = run_agent(env, Crashy(), tmp_path / "run")
    assert metrics["agent_error"] and not metrics["success"] and metrics["n_calls"] == 1


def test_evaluate_runs_and_summary(make_env, tmp_path):
    for preset in ["organized", "horrible"]:
        env, m = make_env("conflict", PRESETS[preset], sweep={"vars": {"preset": preset}, "replicate": 0})
        run_agent(env, OracleAgent(m), tmp_path / "runs" / preset)
    rows = evaluate_runs(tmp_path / "runs", tmp_path / "results.csv")
    assert len(rows) == 2 and (tmp_path / "results.csv").exists()
    summary = summarize(rows)
    assert {s["condition"] for s in summary} == {"preset=organized", "preset=horrible"}
    assert all(s["success"] == 1.0 for s in summary)


def test_answer_parsing_and_normalisation():
    assert parse_answer('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_answer('The answer is {"a": 2} ok') == {"a": 2}
    assert parse_answer("no json") == {}
    assert parse_money("$12,345.60") == 12345.6
    assert parse_money("1234.5 USD") == 1234.5
    assert parse_money(True) is None
    assert str(parse_date("March 11, 2027")) == "2027-03-11"
    assert str(parse_date("03/11/2027")) == "2027-03-11"
