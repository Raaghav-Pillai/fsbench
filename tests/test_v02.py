import json
from copy import deepcopy

import pytest

from fsbench import FSConfig
from fsbench.behavior import trajectory, inspect_run, analyze_behavior
from fsbench.cli import main
from fsbench.config import PRESETS
from fsbench.difficulty import verify_composed
from fsbench.evaluate import evaluate_runs, keep_paired
from fsbench.experiment import build_schedule, fingerprint
from fsbench.generate import generate_env
from fsbench.models import ChatCompletionsAdapter, ModelResponse, create_agent, register_provider
from fsbench.runner import OracleAgent, run_agent
from fsbench.statistics import analyze_statistics, design_pairing
from fsbench.tools import make_toolset


def test_profiles_preserve_facts_and_required_ids(tmp_path):
    envs = []
    for profile in ("clean", "navigation_stress", "ambiguity_stress", "max_stress"):
        env = tmp_path / profile
        generate_env("conflict", PRESETS[profile], env, world_seed=7, task_seed=7, layout_seed=7, label=profile)
        envs.append(env)
    assert verify_composed(envs) == {"worlds": 1, "environments": 4}
    path = envs[-1] / "manifest.json"
    m = json.loads(path.read_text())
    m["files"][0]["document_sha256"] = "changed"
    # Alter a required identity present in every condition, not an optional distractor.
    next(f for f in m["files"] if f["role"] == "required")["document_sha256"] = "changed"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="document contents changed"):
        verify_composed(envs)


def test_cli_factorial_is_default_and_ofat_explicit(tmp_path):
    args = ["sweep", "--task", "conflict", "--vary", "filename_noise=0,0.9",
            "--vary", "stale_version_rate=0,0.5", "--replicates", "1"]
    main(args + ["--out", str(tmp_path / "grid")])
    grid = [json.loads(p.read_text()) for p in (tmp_path / "grid").glob("*/manifest.json")]
    assert len(grid) == 4
    assert all(m["sweep"]["mode"] == "grid" for m in grid)
    verify_composed([p.parent for p in (tmp_path / "grid").glob("*/manifest.json")])
    main(args + ["--out", str(tmp_path / "ofat"), "--ofat"])
    assert all(json.loads(p.read_text())["sweep"]["mode"] == "ofat"
               for p in (tmp_path / "ofat").glob("*/manifest.json"))


def test_adapter_normalizes_missing_usage_and_errors():
    response = ChatCompletionsAdapter(lambda *a: {"choices": [{"message": {"content": "answer"},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}).run([], [], {})
    assert response.content == "answer" and response.finish_reason == "stop"
    assert response.input_tokens == 8 and response.cost_usd is None
    response = ChatCompletionsAdapter(lambda *a: {"error": {"message": "rejected"}}).run([], [], {})
    assert "rejected" in response.error
    assert response.latency_s >= 0


def test_custom_adapter_uses_same_agent_loop(make_env, tmp_path):
    from fsbench.openrouter import OpenRouterAgent
    env, m = make_env("conflict")
    class Adapter:
        def run(self, messages, tools, config):
            return ModelResponse(content=json.dumps(m["ground_truth"]), finish_reason="stop")
    agent = OpenRouterAgent("local/test", adapter=Adapter(), provider="test")
    metrics = run_agent(env, agent, tmp_path / "run")
    assert metrics["success"] and metrics["total_tokens"] is None and metrics["cost_usd"] is None
    assert metrics["usage"]["finish_reasons"] == ["stop"]


def test_provider_failure_recorded(make_env, tmp_path):
    from fsbench.openrouter import OpenRouterAgent
    env, _ = make_env("conflict")
    class Adapter:
        def run(self, *args):
            return ModelResponse(error="provider unavailable", latency_s=.5)
    metrics = run_agent(env, OpenRouterAgent("test", adapter=Adapter()), tmp_path / "run")
    assert metrics["task_failure_type"] == "provider_failure"
    assert metrics["usage"]["llm_calls"] == 1
    assert not metrics["answered_without_evidence"]


def test_multi_model_repeated_runner_and_complete_pairing(tmp_path):
    root = tmp_path / "envs"
    for seed in (7, 8):
        for profile in ("clean", "ambiguity_stress"):
            generate_env("conflict", PRESETS[profile].with_value("distractors", 3),
                         root / f"{seed}_{profile}", world_seed=seed, task_seed=seed, layout_seed=seed, label=profile)
    calls = []
    class Fake:
        def __init__(self, model, **settings):
            self.model, self.name = model, "mock:" + model
        def describe(self):
            return {"model": self.model, "agent_prompt_version": "v1"}
        def run(self, prompt, tools):
            calls.append((self.model, tools.tool_policy))
            return {}
    register_provider("mock_v02", Fake)
    out = tmp_path / "runs"
    args = ["run", "--envs", str(root), "--model", "weak", "--model", "strong", "--provider", "mock_v02",
            "--arm", "files:naturalistic", "--arm", "indexed:index_first", "--trials-per-cell", "2",
            "--interleave", "--order-seed", "7", "--out", str(out), "--skip-existing"]
    main(args)
    assert len(calls) == 32
    rows = evaluate_runs(out, paired=True)
    assert len(rows) == 32
    assert {r["trial_index"] for r in rows} == {1, 2}
    assert len(keep_paired(rows[:-1])) == 16
    bad = deepcopy(rows)
    bad[0]["workspace_sha256"] = "different"
    assert len(keep_paired(bad)) == 16
    assert len(keep_paired(rows + [rows[0]])) == 16
    main(args)
    assert len(calls) == 32
    plan = json.loads((out / "experiment.json").read_text())
    assert len(plan["expected_cells"]) == 16
    assert [c["execution_order"] for c in plan["schedule"]] == list(range(1,33))


def test_schedule_interleaves_trials_and_models(make_env):
    env, _ = make_env("conflict")
    kwargs = dict(models=["a", "b"], arms=[("files", "naturalistic"), ("indexed", "index_first")], trials_per_cell=3)
    a = build_schedule([env], ["files"], **kwargs)
    assert a == build_schedule([env], ["files"], **kwargs)
    assert a != build_schedule([env], ["files"], order_seed=8, **kwargs)
    assert len(a) == 12
    assert len({c["workspace_sha256"] for c in a}) == 1
    with pytest.raises(ValueError, match="positive"):
        build_schedule([env], ["files"], trials_per_cell=0)


def test_world_bootstrap_averages_trials_and_effect_sizes():
    rows = []
    for seed in (1, 2, 3):
        for condition, delta in (("clean", 0), ("stress", 4)):
            for trial, jitter in ((1, -1), (2, 1)):
                rows.append({"task_type": "conflict", "agent": "a", "model": "a", "toolset": "files",
                    "tool_policy": "naturalistic", "condition": condition, "seed.world": seed,
                    "seed.task": seed, "seed.layout": seed, "n_calls": seed + delta + jitter,
                    "success": True, "trial_index": trial})
    report = analyze_statistics(rows)
    contrast = next(r for r in report["contrasts"] if r["metric"] == "n_calls")
    assert contrast["n"] == 3  # Not six independent trials.
    assert contrast["mean_difference"] == contrast["ci_low"] == contrast["ci_high"] == 4
    clean = next(r for r in report["cells"] if r["condition"] == "clean" and r["metric"] == "n_calls")
    assert clean["within_seed_variance"] == 2
    assert clean["between_seed_variance"] == 1
    assert clean["n_trials"] == 6


def test_factorial_interaction_difference_of_differences():
    rows = []
    for seed in (1, 2):
        for x in (0, 1):
            for y in (0, 1):
                rows.append({"task_type": "conflict", "agent": "a", "model": "a", "toolset": "files",
                    "condition": f"x={x};y={y}", "sweep.vars": f"x={x};y={y}", "cfg.x": x, "cfg.y": y,
                    "seed.world": seed, "seed.task": seed, "seed.layout": seed, "n_calls": seed+x+y+5*x*y})
    report = analyze_statistics(rows)
    effect = next(r for r in report["factorial"] if r["metric"] == "n_calls")
    assert effect["mean_difference"] == 5 and effect["n"] == 2


def test_trajectory_read_labels_counts_and_recovery(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(stale_version_rate=1, duplicate_rate=1, distractors=2))
    tools = make_toolset("files", env / "workspace")
    trap = next(f for f in m["files"] if f["role"] == "trap" and f["copy"] == 0)
    req = next(f for f in m["files"] if f["role"] == "required" and f["copy"] == 0)
    copy = next(f for f in m["files"] if f["doc_id"] == req["doc_id"] and f["copy"] == 1)
    tools.call("search_files", {"query": "contract"})
    tools.call("read_file", {"path": "/workspace/" + trap["path"]})
    tools.call("read_file", {"path": "/workspace/missing"})
    tools.call("read_file", {"path": "/workspace/" + req["path"]})
    tools.call("read_file", {"path": "/workspace/" + copy["path"]})
    events, metrics = trajectory(tools.tracer.events, m)
    assert metrics["searches_before_evidence"] == 1
    assert metrics["distractors_before_evidence"] == metrics["stale_reads_before_evidence"] == 1
    assert metrics["duplicate_read_count"] == 1
    assert metrics["recovered_tool_error"]
    assert not metrics["answered_without_evidence"]
    assert "STALE_FILE_READ" in events[1]["labels"]
    assert events[2]["labels"] == ["INVALID_PATH"]
    assert "DUPLICATE_READ" in events[4]["labels"]


def test_inspect_and_batch_analysis(make_env, tmp_path):
    env, m = make_env("conflict")
    run = tmp_path / "run"
    metrics = run_agent(env, OracleAgent(m), run)
    text = inspect_run(run)
    assert "REQUIRED_FILE_READ" in text and "Correct" in text
    assert "FINAL_ANSWER" in text
    main(["inspect", "--run", str(run)])
    main(["analyze-behavior", "--runs", str(run), "--group-by", "model", "toolset"])
    assert (run / "behavior.csv").exists()


def test_profiles_cli_composes_and_verifies(tmp_path):
    root = tmp_path / "profiles"
    main(["sweep", "--task", "conflict", "--profiles", "clean", "ambiguity_stress",
          "--replicates", "2", "--set", "distractors=4", "--out", str(root)])
    main(["validate", "--envs", str(root), "--composed", "--index"])
    assert len(list(root.glob("*/manifest.json"))) == 4


def test_backtracking_proxy_only_counts_unproductive_revisits():
    manifest = {"files": [{"path": "a/required.txt", "doc_id": "r", "role": "required"}], "required_doc_ids": ["r"]}
    def navigate(step, path):
        return {"step": step, "tool": "list_directory", "category": "navigate", "ok": True,
                "args": {"path": path}, "cwd": "/workspace", "read": []}
    events = [navigate(1, "/workspace/a"), navigate(2, "/workspace/b"), navigate(3, "/workspace/a")]
    assert trajectory(events, manifest)[1]["backtrack_count"] == 1
    events += [{"step": 4, "tool": "read_file", "ok": True, "read": ["a/required.txt"]},
               navigate(5, "/workspace/b")]
    assert trajectory(events, manifest)[1]["backtrack_count"] == 1


def test_malformed_calls_are_traced(make_env, tmp_path):
    from test_openrouter import FakeOpenRouter
    env, _ = make_env("conflict")
    agent = FakeOpenRouter([{"tool_calls": [{"id": "c", "function": {"name": "read_file", "arguments": "bad json"}}]},
                           {"content": "{}"}])
    metrics = run_agent(env, agent, tmp_path / "run")
    assert metrics["n_calls"] == metrics["n_errors"] == 1
    assert "not valid JSON" in metrics["trajectory"][0]["error"]


def test_expanded_source_failure_even_without_required_read(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(stale_version_rate=1))
    trap = next(f for f in m["files"] if f["role"] == "trap" and f["doc_id"] in m["decoy_answers"])
    class Stale:
        name = "stale"
        def run(self, prompt, tools):
            tools.call("read_file", {"path": "/workspace/" + trap["path"]})
            return m["decoy_answers"][trap["doc_id"]]
    metrics = run_agent(env, Stale(), tmp_path / "run")
    assert metrics["failure_type"] == "retrieval_failure"  # Legacy API retained.
    assert metrics["task_failure_type"] == "stale_version_confusion"
    assert metrics["failure_family"] == "source_selection_failure"


def test_success_interval_not_point_estimate_for_single_trial():
    rows = [{"task_type": "conflict", "agent": "a", "toolset": "files", "condition": "clean",
             "seed.world": seed, "seed.task": seed, "seed.layout": seed, "success": True} for seed in range(20)]
    r = next(r for r in analyze_statistics(rows)["cells"] if r["metric"] == "success")
    assert .8 < r["ci_low"] < 1 and r["ci_high"] == pytest.approx(1)
