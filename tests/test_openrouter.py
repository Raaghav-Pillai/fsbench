import json

import pytest

from fsbench import FSConfig
from fsbench.openrouter import OpenRouterAgent, load_api_key
from fsbench.runner import run_agent


class FakeOpenRouter(OpenRouterAgent):
    """Replays scripted assistant messages instead of calling the API."""

    def __init__(self, script):
        super().__init__("fake/model", api_key="test", max_steps=5)
        self.script = list(script)
        self.requests = []

    def _post(self, path, body):
        self.requests.append(json.loads(json.dumps(body)))
        msg = self.script.pop(0)
        return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}


def _call(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_loop_executes_tools_and_is_scored(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(mime_types=("txt",)))
    req = next(f["path"] for f in m["files"] if f["role"] == "required")
    gt = m["ground_truth"]
    agent = FakeOpenRouter([
        {"content": None, "tool_calls": [_call(1, "list_directory", {"path": "/workspace"})]},
        {"content": "", "tool_calls": [_call(2, "read_file", {"path": f"/workspace/{req}"}),
                                       _call(3, "read_file", {"path": "/workspace/nope.txt"})]},
        {"content": "```json\n" + json.dumps(gt) + "\n```"},
    ])
    metrics = run_agent(env, agent, tmp_path / "run", toolset="files")

    assert metrics["success"]
    assert metrics["n_calls"] == 3
    assert metrics["path_hallucination_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["usage"]["llm_calls"] == 3
    assert metrics["usage"]["cost_usd"] == pytest.approx(0.003)
    assert {t["function"]["name"] for t in agent.requests[0]["tools"]} >= {"read_file", "write_file"}
    tool_msgs = [msg for msg in agent.transcript if msg["role"] == "tool"]
    assert [t["tool_call_id"] for t in tool_msgs] == ["c1", "c2", "c3"]
    assert (tmp_path / "run" / "transcript.json").exists()


def test_step_limit_forces_final_answer(make_env, tmp_path):
    env, _ = make_env("retrieve")
    loop = [{"content": "", "tool_calls": [_call(i, "list_directory", {})]} for i in range(5)]
    agent = FakeOpenRouter(loop + [{"content": '{"employee_id": "E-1"}'}])
    metrics = run_agent(env, agent, tmp_path / "run")
    assert metrics["usage"]["hit_step_limit"]
    assert "tools" not in agent.requests[-1]
    assert not metrics["success"]


def test_api_key_lookup(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No OpenRouter API key"):
        load_api_key(tmp_path / ".env")
    (tmp_path / ".env").write_text('OTHER=1\nOPENROUTER_API_KEY="sk-or-abc"\n')
    assert load_api_key(tmp_path / ".env") == "sk-or-abc"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert load_api_key(tmp_path / ".env") == "sk-or-env"
