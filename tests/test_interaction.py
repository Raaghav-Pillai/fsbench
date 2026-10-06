import json

import pytest

from fsbench.config import FSConfig
from fsbench.docs import ALL_FORMATS, Doc
from fsbench.evaluate import balance_warnings, evaluate_runs, keep_paired
from fsbench.experiment import build_schedule, save_plan, validate_environment, workspace_inventory
from fsbench.interaction import bootstrap_difference, compare_trace, interaction_tables, write_interaction
from fsbench.render import render
from fsbench.runner import OracleAgent, run_agent
from fsbench.tools import IndexedTools, make_toolset


@pytest.fixture
def indexed_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    for fmt in ALL_FORMATS:
        doc = Doc(fmt, "contract", "needleword shared content", "same", (),
                  fields=[("Identifier", "needle" + fmt), ("Status", "Executed")])
        (ws / f"document.{fmt}").write_bytes(render(doc, fmt))
    (tmp_path / "manifest.json").write_text('{"secret": "forbiddenmetadataneedle"}')
    (tmp_path / "answer.json").write_text("forbiddenmetadataneedle")
    return ws


def test_index_covers_every_format_and_returns_readable_paths(indexed_workspace):
    tool = IndexedTools(indexed_workspace)
    out = tool.call("search_index", {"query": "needleword", "k": 100})
    paths = tool.tracer.events[-1]["seen"]
    assert set(paths) == {f"document.{fmt}" for fmt in ALL_FORMATS}
    assert set(tool._index["docs"]) == set(paths)
    for p in paths:
        assert f"/workspace/{p}" in out
        assert "needleword" in tool.call("read_file", {"path": f"/workspace/{p}"})
    for fmt in ALL_FORMATS:
        tool.call("search_index", {"query": "needle" + fmt})
        assert tool.tracer.events[-1]["seen"] == [f"document.{fmt}"]


@pytest.mark.parametrize("toolset", ["shell", "files", "indexed"])
def test_hidden_metadata_is_inaccessible(indexed_workspace, toolset):
    tool = make_toolset(toolset, indexed_workspace)
    read = "cat" if toolset == "shell" else "read_file"
    for path in ("../manifest.json", "/manifest.json", "/workspace/../answer.json", "/workspace/manifest.json"):
        assert tool.call(read, {"path": path}).startswith("Error:")
    if toolset == "shell":
        listing = tool.call("find", {"type": "f"})
        result = tool.call("grep", {"pattern": "forbiddenmetadataneedle"})
    else:
        listing = tool.call("glob", {"pattern": "**/*"})
        result = tool.call("search_files", {"query": "forbiddenmetadataneedle"})
    assert "manifest" not in listing and "answer.json" not in listing
    assert result == "(no matches)"
    if toolset == "indexed":
        assert tool.call("search_index", {"query": "forbiddenmetadataneedle"}) == "(no matches)"


def test_index_independent_of_hidden_labels_and_deterministic(indexed_workspace):
    a = IndexedTools(indexed_workspace)
    before = a.call("search_index", {"query": "needleword executed"})
    (indexed_workspace.parent / "manifest.json").write_text('{"required_doc_ids": ["docx"], "roles": "swapped"}')
    b = IndexedTools(indexed_workspace)
    assert b.call("search_index", {"query": "needleword executed"}) == before
    assert b._index == a._index


def test_new_noise_environment_cannot_reuse_old_index(make_env):
    env, m = make_env("conflict", FSConfig(filename_noise=0))
    other, noisy = make_env("conflict", FSConfig(filename_noise=1))
    a, b = IndexedTools(env / "workspace"), IndexedTools(other / "workspace")
    a.call("search_index", {"query": "agreement"})
    b.call("search_index", {"query": "agreement"})
    assert set(b._index["docs"]) == {f["path"] for f in noisy["files"]}
    assert not set(a._index["docs"]) & set(b._index["docs"])
    for p in b.tracer.events[-1]["seen"]:
        assert not b.call("read_file", {"path": "/workspace/" + p}).startswith("Error:")


def test_index_write_invalidation_and_extraction_errors(indexed_workspace, monkeypatch):
    tool = IndexedTools(indexed_workspace)
    tool.call("search_index", {"query": "needleword"})
    tool.call("write_file", {"path": "fresh.txt", "content": "freshuniqueword"})
    assert "fresh.txt" in tool.call("search_index", {"query": "freshuniqueword"})
    assert "Error:" in tool.call("search_index", {"query": "needleword", "k": -1})
    def broken(path):
        raise ValueError("broken extractor")
    monkeypatch.setattr("fsbench.tools.extract_text", broken)
    broken_tool = IndexedTools(indexed_workspace)
    assert "broken extractor" in broken_tool.call("search_index", {"query": "needleword"})
    assert broken_tool._index is None


def test_scheduling_and_identical_trial_inputs(make_env, tmp_path):
    envs = [make_env("conflict", FSConfig(filename_noise=n), seed=s)[0]
            for s in (1, 2) for n in (0, .3, .6, .9)]
    plan = build_schedule(envs, ["shell", "files", "indexed"], check_index=True)
    assert plan == build_schedule(list(reversed(envs)), ["indexed", "files", "shell"])
    assert plan != build_schedule(envs, ["shell", "files", "indexed"], order_seed=8)
    assert [c["execution_order"] for c in plan] == list(range(1, 25))
    assert {c["world_seed"] for c in plan[:12]} == {1}
    assert {c["toolset"] for c in plan[:12]} == {"shell", "files", "indexed"}
    assert len({(c["filename_noise"], c["toolset"]) for c in plan[:12]}) == 12
    target = tmp_path / "plan.json"
    save_plan(target, {"schedule": plan})
    save_plan(target, {"schedule": plan})
    with pytest.raises(ValueError, match="differs"):
        save_plan(target, {"schedule": plan[::-1]})
    env = envs[0]
    m = json.loads((env / "manifest.json").read_text())
    original = workspace_inventory(env / "workspace")
    hashes = set()
    for ts in ("shell", "files", "indexed"):
        cell = next(c for c in plan if c["env_dir"] == str(env.resolve()) and c["toolset"] == ts)
        run_agent(env, OracleAgent(m), tmp_path / ts, toolset=ts, run_context={**cell, "experiment": "test"})
        assert workspace_inventory(tmp_path / ts / "workspace") == original
        meta = json.loads((tmp_path / ts / "meta.json").read_text())
        hashes.add(meta["workspace_sha256"])
        assert meta["execution_order"] == cell["execution_order"]
        assert meta["experiment_order_seed"] == 7
    assert len(hashes) == 1
    assert workspace_inventory(env / "workspace") == original


def factorial_rows():
    cells = [{"filename_noise": n, "toolset": t} for n in (0, .3, .6, .9)
             for t in ("shell", "files", "indexed")]
    return [{**cell, "expected_cells": json.dumps(cells), "experiment": "test", "agent": "a",
             "seed.world": s, "seed.task": s, "seed.layout": s, "input_sha256": f"input-{s}-{cell['filename_noise']}",
             "workspace_sha256": f"workspace-{s}-{cell['filename_noise']}", "success": True,
             "n_calls": 2 + cell["filename_noise"] * {"shell": 4, "files": 2, "indexed": 1}[cell["toolset"]],
             "files_read": 1, "total_tokens": 100, "cost_usd": .001, "wall_time_s": 1}
            for s in (1, 2) for cell in cells]


def test_pairing_drops_whole_seed_for_missing_duplicate_and_mismatched_cells():
    rows = factorial_rows()
    assert len(keep_paired(rows)) == 24
    partial = rows[:-1]
    assert len(keep_paired(partial)) == 12
    assert {r["seed.world"] for r in keep_paired(partial)} == {1}
    assert any("missing cells" in w and "indexed" in w for w in balance_warnings(partial))
    assert len(keep_paired(rows + [rows[-1]])) == 12
    changed = [dict(r) for r in rows]
    changed[-1]["input_sha256"] = "different"
    assert len(keep_paired(changed)) == 12
    changed[-1]["seed.layout"] = 99
    assert len(keep_paired(changed)) == 12
    missing_toolset = [r for r in rows if r["toolset"] != "indexed"]
    assert keep_paired(missing_toolset) == []


def test_bootstrap_and_interaction_contrasts(tmp_path):
    assert bootstrap_difference([1, 2, 3]) == bootstrap_difference([1, 2, 3])
    assert bootstrap_difference([2, 2])["ci_low"] == 2
    assert bootstrap_difference([-1, 1])["includes_zero"]
    tables = interaction_tables(factorial_rows())
    delta = next(r for r in tables["within_toolset"] if r["toolset"] == "shell" and r["metric"] == "n_calls")
    assert delta["mean_difference"] == pytest.approx(3.6)
    contrast = next(r for r in tables["interaction"] if r["toolset_a"] == "files"
                    and r["toolset_b"] == "shell" and r["metric"] == "n_calls")
    assert contrast["mean_difference"] == pytest.approx(1.8)
    assert len(tables["between_toolsets"]) == 3 * 4 * 6
    report = write_interaction(factorial_rows(), tmp_path)
    assert "At noise 0.9" in report.read_text()
    assert (tmp_path / "sensitivity.csv").exists()
    missing_tokens = [{**r, "total_tokens": None} for r in factorial_rows()]
    assert "missing" in write_interaction(missing_tokens, tmp_path / "missing").read_text()


def test_trace_annotations(make_env, tmp_path):
    from test_evaluate import ScriptedAgent
    env, m = make_env("conflict", FSConfig(filename_noise=.9))
    required = next(f["path"] for f in m["files"] if f["role"] == "required")
    trap = next(f["path"] for f in m["files"] if f["role"] == "trap")
    calls = [("read_file", {"path": "missing.txt"}), ("read_file", {"path": trap}),
             ("read_file", {"path": required})]
    run_agent(env, ScriptedAgent(calls, m["ground_truth"]), tmp_path / "runs" / "files")
    out = compare_trace(tmp_path / "runs", 7, .9)
    assert "REQUIRED EVIDENCE" in out and "STALE/DRAFT TRAP" in out and "INVALID PATH" in out
    assert "Required evidence at step 3" in out


def test_preflight_rejects_modified_or_extra_files(make_env):
    env, m = make_env("conflict")
    validate_environment(env, check_index=True)
    task_path = env / "task.json"
    original_task = task_path.read_bytes()
    task_path.write_text('{"prompt": "different task"}')
    with pytest.raises(ValueError, match="agent task"):
        validate_environment(env)
    task_path.write_bytes(original_task)
    path = env / "workspace" / m["files"][0]["path"]
    original = path.read_bytes()
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        validate_environment(env)
    path.write_bytes(original)
    (env / "workspace" / "answer.json").write_text("hidden answer")
    with pytest.raises(ValueError, match="inventory"):
        validate_environment(env)


def test_multi_toolset_cli_records_plan_and_resumes(make_env, tmp_path, monkeypatch):
    from fsbench.cli import main

    # Put exactly these four existing environments under a common experiment root.
    root = tmp_path / "environments"
    from fsbench.generate import generate_env
    for n in (0, .3, .6, .9):
        generate_env("conflict", FSConfig(filename_noise=n), root / str(n),
                     world_seed=7, task_seed=7, layout_seed=7)
    calls = []
    class FakeAgent:
        def __init__(self, model, max_steps, temperature):
            self.name = "fake:" + model
            self.model, self.max_steps, self.temperature = model, max_steps, temperature
        def describe(self):
            return {"model": self.model, "max_steps": self.max_steps,
                    "temperature": self.temperature, "agent_prompt_version": "v1"}
        def run(self, prompt, tools):
            calls.append(tools.name)
            return {}
    monkeypatch.setattr("fsbench.openrouter.OpenRouterAgent", FakeAgent)
    out = tmp_path / "runs"
    args = ["run", "--envs", str(root), "--out", str(out), "--model", "test/model",
            "--toolset", "shell", "files", "indexed", "--skip-existing"]
    main(args)
    assert len(calls) == 12
    plan = json.loads((out / "experiment.json").read_text())
    assert calls == [c["toolset"] for c in plan["schedule"]]
    rows = evaluate_runs(out, out / "results.csv", paired=True)
    assert len(rows) == 12
    assert {r["prompt_version"] for r in rows} == {"v1"}
    assert {r["experiment"] for r in rows} == {"filename_noise_toolset_interaction_v1"}
    assert {r["execution_order"] for r in rows} == set(range(1, 13))
    main(args)
    assert len(calls) == 12  # No duplicate model calls on resume.
