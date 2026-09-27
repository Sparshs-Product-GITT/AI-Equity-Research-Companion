from pathlib import Path

from ..context import RunContext
from ..schemas import CompanyOverviewOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "company_overview"
TITLE = "Company & business overview"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: Business Overview. Facts only — what the company sells, to whom, how it earns revenue, and how revenue splits by segment and geography. No opinions on competitive position, moat, or outlook; another section owns judgment.

Method:
1. Start from the 10-K Item 1 excerpt provided. It is the primary source for the description, business model, products and customer base.
2. Use WebSearch/WebFetch only to fill what the excerpt lacks: segment revenue percentages (10-K segment note or MD&A), geographic mix, recurring-revenue share. The latest 10-K on sec.gov and the investor-relations site are the preferred sources.
3. Segment percentages must sum to roughly 100 (allow rounding); if only absolute dollars are disclosed, compute the percentage and note the fiscal year.
4. `one_liner` is written for a header strip: <= 25 words, plain English, no adjectives like 'leading'.
5. `sub_industry` must be a precise label an analyst would use (e.g. 'Enterprise application software (CRM)', 'Digital advertising platform', 'Infrastructure software / cloud data')."""


def build_prompt(ctx: RunContext, feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        f"10-K filing: {ctx.tenk.url}",
        "",
        "=== 10-K Item 1 (Business) excerpt ===",
        ctx.tenk.business or "(section not extracted; fetch the 10-K yourself)",
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
        output_model=CompanyOverviewOutput,
        max_turns=20,
        cwd=cwd,
        on_event=on_event,
    )
