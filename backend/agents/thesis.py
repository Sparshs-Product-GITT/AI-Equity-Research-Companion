import json
from pathlib import Path
from typing import Any

from ..context import RunContext
from ..schemas import ThesisOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "thesis_synthesis"
TITLE = "Investment thesis — key debate, bull and bear cases"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: Investment Thesis. You have NO research tools. You synthesize strictly from the section outputs provided; you may not introduce any fact, number, or claim that is not in them.

Deliver:
- `key_debate`: one sentence naming the central tension an investor must resolve (e.g. durability of growth vs. the multiple, margin expansion vs. AI capex, platform consolidation vs. best-of-breed competition). No rating, no target price, no 'we recommend'.
- `bull_case`: 3-4 bullets. Each <= 30 words, each anchored to a specific figure or sourced fact from the inputs (cite the number inline, e.g. 'FCF margin expanded to 34.5% TTM from 27.2% in FY2024').
- `bear_case`: 3-4 bullets, same discipline. Include the valuation angle (rich or cheap vs. peers and vs. own 3-year history) on whichever side the numbers support, and reference the most decision-relevant risk from the Catalysts & Risks section without re-listing all of them.

Write like a seasoned analyst: concrete, balanced, no hedging filler, no adjectives without a number behind them. The page already footnotes every section's sources, so do NOT include URLs, filing names, or 'Sources:' anywhere in the key debate or bullets — the schema rejects text over the limit."""


def build_prompt(ctx: RunContext, sections: dict[str, Any], feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        "=== Computed financials & valuation (ground truth) ===",
        ctx.financials_brief(),
        "",
        "=== Peer comparison (computed) ===",
        json.dumps(sections.get("peer_metrics", {}), indent=1),
        "",
        "=== Section outputs from the research subagents ===",
    ]
    for key in ("company_overview", "financials_enrichment", "catalysts_risks", "news_digest"):
        parts += [f"--- {key} ---", json.dumps(sections.get(key) or {"status": "unavailable"}, indent=1)]
    if feedback:
        parts += ["", "Reviewer feedback on your previous attempt — fix these:", feedback]
    return "\n".join(parts)


async def run(
    ctx: RunContext,
    sections: dict[str, Any],
    cwd: Path,
    on_event: EventCallback | None = None,
    feedback: str | None = None,
) -> AgentRun:
    return await run_structured_agent(
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_prompt(ctx, sections, feedback),
        output_model=ThesisOutput,
        tools=(),
        max_turns=4,
        cwd=cwd,
        on_event=on_event,
    )
