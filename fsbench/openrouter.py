"""Tool-calling agent backed by any model on OpenRouter (OpenAI-compatible chat API).

The API key is read from the ``OPENROUTER_API_KEY`` environment variable, or from a
``.env`` file in the current directory containing ``OPENROUTER_API_KEY=sk-or-...``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from fsbench.tools import ToolSet

BASE_URL = "https://openrouter.ai/api/v1"
AGENT_PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an autonomous agent working on a company file share. Use the provided tools to "
    "explore the files and gather evidence; do not guess file paths or contents. When you have "
    "the answer, stop calling tools and reply with only the requested JSON object."
)


def load_api_key(env_file: str | Path = ".env") -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    path = Path(env_file)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "No OpenRouter API key found. Set the OPENROUTER_API_KEY environment variable or add "
        "a line OPENROUTER_API_KEY=sk-or-... to a .env file in the project folder."
    )


class OpenRouterAgent:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_steps: int = 40,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = BASE_URL,
        timeout_s: float = 180.0,
        max_retries: int = 4,
    ):
        self.model = model
        self.name = f"openrouter:{model}"
        self.api_key = api_key or load_api_key()
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.usage: dict = {}
        self.transcript: list[dict] = []

    def describe(self) -> dict:
        return {
            "kind": "openrouter",
            "model": self.model,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "agent_prompt_version": AGENT_PROMPT_VERSION,
            "system_prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        }

    def run(self, prompt: str, tools: ToolSet) -> str:
        self.usage = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
            "llm_calls": 0, "hit_step_limit": False, "llm_latency_s": 0.0, "llm_call_latencies": [],
        }
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        self.transcript = messages
        specs = tools.specs("openai")
        last_text = ""
        for _ in range(self.max_steps):
            msg = self._complete(messages, specs)
            last_text = msg.get("content") or last_text
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if not calls:
                return last_text
            for call in calls:
                fn = call.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    result = tools.call(fn.get("name", ""), args if isinstance(args, dict) else {})
                except json.JSONDecodeError as e:
                    result = f"Error: tool arguments were not valid JSON ({e})"
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
        self.usage["hit_step_limit"] = True
        messages.append({"role": "user", "content": "Step limit reached. Reply now with only your final JSON answer."})
        msg = self._complete(messages, specs, allow_tools=False)
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
        return msg.get("content") or last_text

    def _complete(self, messages: list[dict], specs: list[dict], allow_tools: bool = True) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        if allow_tools:
            body["tools"] = specs
        t0 = time.perf_counter()
        data = self._post("/chat/completions", body)
        dt = time.perf_counter() - t0
        self.usage["llm_latency_s"] = round(self.usage["llm_latency_s"] + dt, 4)
        self.usage["llm_call_latencies"].append(round(dt, 4))
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        usage = data.get("usage") or {}
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += usage.get("prompt_tokens", 0) or 0
        self.usage["output_tokens"] += usage.get("completion_tokens", 0) or 0
        self.usage["total_tokens"] = self.usage["input_tokens"] + self.usage["output_tokens"]
        self.usage["cost_usd"] = round(self.usage["cost_usd"] + float(usage.get("cost", 0) or 0), 6)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)[:500]}")
        return choices[0].get("message") or {}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "fsbench",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:1000]
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"OpenRouter request failed: {e}") from None
        raise AssertionError("unreachable")
