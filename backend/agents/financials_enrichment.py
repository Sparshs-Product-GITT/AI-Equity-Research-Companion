from pathlib import Path

from ..context import RunContext
from ..schemas import FinancialsEnrichmentOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "financials_enrichment"
TITLE = "Financial performance — disclosed SaaS metrics"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: the non-GAAP operating metrics that XBRL does not tag. The GAAP financials (revenue, margins, FCF, EPS) have ALREADY been computed deterministically from SEC XBRL data and are given to you for context — do not re-derive or contradict them.

Find only metrics the company itself discloses, from the latest earnings press release, 10-K/10-Q, or investor presentation:
- Remaining performance obligations (RPO) and current RPO
- Annual recurring revenue (ARR) or subscription & support revenue growth
- Net revenue retention / dollar-based net expansion
- Billings or bookings growth
- Customer counts, large-customer counts (e.g. >$1M ARR), paid seats, if disclosed

Rules:
- Report each metric exactly as disclosed, with the period it refers to and the URL.
- If a metric is not disclosed, do not include it. An empty `saas_metrics` list is a valid answer.
- `reporting_notes`: 0-3 items of accounting context a reader needs (fiscal-year convention, a large acquisition distorting growth, a one-off tax benefit inflating net margin, restated segments). Only include what you can source."""


def build_prompt(ctx: RunContext, feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        "=== Computed GAAP financials (from SEC XBRL; treat as ground truth) ===",
        ctx.financials_brief(),
        "=== end ===",
        "",
        f"Latest 10-K: {ctx.tenk.url}",
        "",
        "=== 10-K Item 7 (MD&A) excerpt — often contains RPO, cRPO, retention, and segment commentary ===",
        ctx.tenk.mdna[:25_000] or "(not extracted)",
        "=== end excerpt ===",
    ]
    if feedback:
        parts += ["", "Reviewer feedback on your previous attempt — fix these:", feedback]
    return "\n".join(parts)


async def run(ctx: RunContext, cwd: Path, on_event: EventCallback | None = None, feedback: str | None = None) -> AgentRun:
    return await run_structured_agent(
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_prompt(ctx, feedback),
        output_model=FinancialsEnrichmentOutput,
        max_turns=20,
        cwd=cwd,
        on_event=on_event,
    )
