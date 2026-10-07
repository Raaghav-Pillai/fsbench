"""Provider adapters and a registry. Benchmark scheduling never interprets model names."""
from dataclasses import dataclass, field
import os
import time
from typing import Callable, Protocol


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None


class ModelAdapter(Protocol):
    def run(self, messages: list[dict], tools: list[dict], config: dict) -> ModelResponse: ...


class ChatCompletionsAdapter:
    """Normalize an OpenAI-compatible endpoint; transport owns auth and retries."""
    def __init__(self, post: Callable):
        self.post = post

    def run(self, messages, tools, config):
        body = {**config, "messages": messages}
        if tools:
            body["tools"] = tools
        start = time.perf_counter()
        try:
            data = self.post("/chat/completions", body)
        except Exception as e:
            return ModelResponse(latency_s=time.perf_counter() - start, error=str(e))
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        error = str(data["error"]) if data.get("error") else None
        if not data.get("choices") and error is None:
            error = "provider returned no choices"
        return ModelResponse(content=msg.get("content") or "", tool_calls=msg.get("tool_calls") or [],
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"), latency_s=time.perf_counter() - start,
            finish_reason=choice.get("finish_reason"), error=error)


AGENT_FACTORIES: dict[str, Callable] = {}


def register_provider(name: str, factory: Callable) -> None:
    """Register factory(model, max_steps, temperature) -> agent implementing runner.Agent."""
    AGENT_FACTORIES[name] = factory


def create_agent(provider: str, model: str, **settings):
    if provider not in AGENT_FACTORIES:
        raise ValueError(f"unknown provider {provider!r}; registered: {sorted(AGENT_FACTORIES)}")
    return AGENT_FACTORIES[provider](model, **settings)


def _openrouter(model, **settings):
    from fsbench.openrouter import OpenRouterAgent
    return OpenRouterAgent(model, **settings)


def _compatible(model, **settings):
    from fsbench.openrouter import OpenRouterAgent
    url = os.environ.get("FSBENCH_BASE_URL")
    key = os.environ.get("FSBENCH_API_KEY")
    if not url or not key:
        raise ValueError("compatible provider requires FSBENCH_BASE_URL and FSBENCH_API_KEY")
    return OpenRouterAgent(model, base_url=url, api_key=key, provider="compatible", **settings)


register_provider("openrouter", _openrouter)
register_provider("compatible", _compatible)
