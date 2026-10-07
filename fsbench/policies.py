"""Explicit interventions on the agent interface, independent of task scoring."""

POLICIES = ("naturalistic", "require_index", "index_first", "files_only", "shell_only")


def validate_policy(toolset: str, policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unknown tool policy: {policy}")
    expected = {"require_index": "indexed", "index_first": "indexed",
                "files_only": "files", "shell_only": "shell"}.get(policy)
    if expected and toolset != expected:
        raise ValueError(f"{policy} requires toolset={expected}, got {toolset}")


def required_tools(policy: str) -> list[str]:
    return ["search_index"] if policy in ("require_index", "index_first") else []


def instructions(policy: str) -> str:
    return {
        "require_index": "You must successfully call search_index at least once before your final answer.",
        "index_first": "Begin with search_index. After a successful indexed search, the remaining file tools become available. You must use the index before reading files or answering.",
    }.get(policy, "")


def usage_metrics(events: list[dict], policy: str) -> dict:
    from collections import Counter
    counts = Counter(e["tool"] for e in events)
    required = required_tools(policy)
    successful = {e["tool"] for e in events if e["ok"]}
    first = next((i for i, e in enumerate(events) if e["tool"] in required and e["ok"]), None)
    return {"tool_usage_count_by_name": dict(counts), "tool_usage": dict(counts),
            "first_tool_used": events[0]["tool"] if events else None,
            "steps_before_required_tool": first,
            "required_tool_used": set(required).issubset(successful),
            "required_tools": required, "tools_used": sorted(counts)}
