import json

import pytest

from fsbench.runner import run_agent
from fsbench.tools import make_toolset
from test_openrouter import FakeOpenRouter, _call


def test_index_first_is_enforced_and_failed_search_does_not_unlock(tmp_path):
    (tmp_path / "a.txt").write_text("contract terms")
    tools = make_toolset("indexed", tmp_path, tool_policy="index_first")
    assert tools.available == ("search_index",)
    assert tools.call("read_file", {"path": "/workspace/a.txt"}).startswith("Error")
    assert not tools.tracer.events[-1]["read"]
    tools.call("search_index", {"query": "contract", "k": 0})
    assert tools.available == ("search_index",)
    tools.call("search_index", {"query": "contract"})
    assert "read_file" in tools.available
    assert tools.call("read_file", {"path": "/workspace/a.txt"}) == "contract terms"


@pytest.mark.parametrize("policy", ["require_index", "index_first"])
def test_correct_answer_without_required_tool_is_invalid(make_env, tmp_path, policy):
    env, m = make_env("conflict")
    agent = FakeOpenRouter([{"content": json.dumps(m["ground_truth"])}])
    metrics = run_agent(env, agent, tmp_path / "run", toolset="indexed", tool_policy=policy)
    assert metrics["answer_correct"] and metrics["score"] == 1
    assert not metrics["success"] and not metrics["valid"]
    assert metrics["failure_type"] == "required_tool_not_used"
    assert metrics["n_calls"] == 0


def test_dynamic_schemas_and_policy_metadata(make_env, tmp_path):
    env, m = make_env("conflict")
    required = next(f for f in m["files"] if f["role"] == "required")
    agent = FakeOpenRouter([
        {"tool_calls": [_call(1, "search_index", {"query": "contract"})]},
        {"tool_calls": [_call(2, "read_file", {"path": "/workspace/" + required["path"]})]},
        {"content": json.dumps(m["ground_truth"])}])
    metrics = run_agent(env, agent, tmp_path / "run", toolset="indexed", tool_policy="index_first")
    assert [t["function"]["name"] for t in agent.requests[0]["tools"]] == ["search_index"]
    assert "read_file" in [t["function"]["name"] for t in agent.requests[1]["tools"]]
    assert metrics["success"] and metrics["required_tool_used"]
    assert metrics["steps_before_required_tool"] == 0
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["required_tools"] == ["search_index"]
    assert meta["tool_usage_count_by_name"] == {"read_file": 1, "search_index": 1}


@pytest.mark.parametrize("toolset,policy", [("files", "index_first"), ("indexed", "files_only"), ("files", "shell_only")])
def test_incompatible_policies_rejected(tmp_path, toolset, policy):
    with pytest.raises(ValueError):
        make_toolset(toolset, tmp_path, tool_policy=policy)
