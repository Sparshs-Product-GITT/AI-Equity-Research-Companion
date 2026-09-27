"""Thin wrapper around the Claude Agent SDK: one sandboxed `query()` per subagent, with a
restricted tool set, a structured-output schema, a spend cap, and an event stream for the UI."""
import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)
from pydantic import BaseModel, ValidationError

from ..config import ANTHROPIC_ENV, CLI_PATH, MAX_BUDGET_USD, MODEL

EventCallback = Callable[[str, str, dict[str, Any]], Awaitable[None] | None]

ANALYST_PREAMBLE = """You are a senior sell-side equity research analyst covering US-listed large-cap software and internet companies. You are producing one precisely scoped section of a one-page pre-read for another analyst who will write the full report.

Working rules:
- Prefer primary sources: SEC filings on sec.gov, the company's investor-relations site, earnings press releases and call transcripts. Use reputable financial press only to fill gaps.
- Never invent a number, date, or quote. If something cannot be found, say so explicitly in the output rather than guessing.
- Every fact you report must be traceable to a URL you actually opened (WebFetch) or that appeared in your search results (WebSearch). Put those URLs in `sources`.
- This is a ONE-PAGER, not a memo. Every text field has a hard character limit in the schema and the output is rejected if you exceed it. Write telegraphic analyst notes: one idea per field, numbers over words, no hedging filler, no parenthetical caveats longer than a clause.
- Never put URLs, document names, or 'Sources:' text inside narrative fields. Provenance goes only in the `sources` list (or a `source` object) — the page renders it as a footnote.
- Sentence case. No marketing language. Numbers beat adjectives.
- Do the research efficiently: 4-10 tool calls is typical. Stop when you have what the schema asks for.
- Your final message must be the structured output requested and nothing else."""


@dataclass
class AgentRun:
    name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    raw_text: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None


async def _emit(cb: EventCallback | None, name: str, kind: str, detail: dict[str, Any]) -> None:
    if cb is None:
        return
    r = cb(name, kind, detail)
    if asyncio.iscoroutine(r):
        await r


async def run_structured_agent(
    *,
    name: str,
    system_prompt: str,
    prompt: str,
    output_model: type[BaseModel],
    tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    max_turns: int = 30,
    cwd: Path | None = None,
    on_event: EventCallback | None = None,
    model: str | None = None,
) -> AgentRun:
    started = time.monotonic()
    run = AgentRun(name=name, ok=False)
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=list(tools),
        allowed_tools=list(tools),
        model=model or MODEL,
        max_turns=max_turns,
        max_budget_usd=MAX_BUDGET_USD,
        setting_sources=[],
        cwd=str(cwd) if cwd else None,
        output_format={"type": "json_schema", "schema": output_model.model_json_schema()},
        env={"CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "0", **ANTHROPIC_ENV},
        cli_path=CLI_PATH,
    )
    await _emit(on_event, name, "started", {})
    result: ResultMessage | None = None
    texts: list[str] = []
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        summary = _summarize_tool_input(block.name, block.input)
                        run.tool_calls.append({"tool": block.name, "input": summary})
                        await _emit(on_event, name, "tool", {"tool": block.name, "input": summary})
                    elif isinstance(block, TextBlock) and block.text.strip():
                        texts.append(block.text)
            elif isinstance(message, ResultMessage):
                result = message
    except Exception as exc:  # the SDK raises after yielding an error ResultMessage
        if result is None:
            run.error = f"{type(exc).__name__}: {exc}"
    run.raw_text = "\n\n".join(texts)
    run.duration_ms = int((time.monotonic() - started) * 1000)
    if result is not None:
        run.cost_usd = result.total_cost_usd or 0.0
        run.num_turns = result.num_turns or 0
        run.session_id = result.session_id
        run.duration_ms = result.duration_ms or run.duration_ms
        if result.subtype == "success" and result.structured_output:
            try:
                run.data = output_model.model_validate(result.structured_output).model_dump(mode="json")
                run.ok = True
            except ValidationError as ve:
                run.error = f"Schema validation failed: {ve.errors()[:3]}"
        else:
            errs = "; ".join(str(e) for e in (result.errors or [])) or (result.result or "")
            run.error = f"{result.subtype}: {errs}".strip(": ")
    await _emit(
        on_event,
        name,
        "finished" if run.ok else "failed",
        {"cost_usd": round(run.cost_usd, 4), "turns": run.num_turns, "error": run.error},
    )
    return run


def _summarize_tool_input(tool: str, inp: dict[str, Any]) -> str:
    if tool == "WebSearch":
        return inp.get("query", "")
    if tool == "WebFetch":
        return inp.get("url", "")
    return ", ".join(f"{k}={str(v)[:60]}" for k, v in inp.items())[:200]


def to_markdown(title: str, data: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in data.items():
        lines.append(f"## {key.replace('_', ' ').capitalize()}")
        lines.extend(_md_value(value, 0))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _md_value(value: Any, depth: int) -> list[str]:
    pad = "  " * depth
    if isinstance(value, dict):
        if set(value) == {"title", "url"}:
            return [f"{pad}- [{value['title']}]({value['url']})"]
        out = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.append(f"{pad}- **{k.replace('_', ' ')}**:")
                out.extend(_md_value(v, depth + 1))
            else:
                out.append(f"{pad}- **{k.replace('_', ' ')}**: {v if v is not None else 'n/a'}")
        return out
    if isinstance(value, list):
        if not value:
            return [f"{pad}- none"]
        out = []
        for item in value:
            if isinstance(item, (dict, list)):
                out.extend(_md_value(item, depth))
                out.append("")
            else:
                out.append(f"{pad}- {item}")
        return out
    return [f"{pad}{value if value is not None else 'n/a'}"]
