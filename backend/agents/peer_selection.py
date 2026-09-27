from pathlib import Path

from ..context import RunContext
from ..schemas import PeerSelectionOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "peer_selection"
TITLE = "Valuation — peer group selection"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: choose the peer group. You do NOT compute multiples — the pipeline computes trailing EV/Revenue, EV/EBITDA, P/FCF, P/E and revenue growth for each peer you name, straight from SEC XBRL and market prices.

Pick 3 to 5 direct comparables a sell-side analyst would put in this company's comp table:
- Same sub-industry and business model (e.g. enterprise application SaaS vs. infrastructure software vs. ad-supported internet). Do not mix models.
- Similar scale where possible (large-cap), so multiples are comparable.
- MUST be US-domestic SEC filers that file 10-K/10-Q (not 20-F/40-F foreign private issuers), because peer financials are pulled from us-gaap XBRL. Exclude ADRs.
- Use the company's own 10-K competition discussion, sell-side comp tables quoted in the press, and the company's proxy peer group as evidence; cite them.
- Return the exact US ticker symbol. Double-check the symbol maps to the intended company.
- `selection_rationale` is ONE sentence naming the screen (sub-industry, business model, scale band). Exclusions and caveats do not belong on the page — leave them out."""


def build_prompt(ctx: RunContext, feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        "Business context (from the 10-K Item 1 excerpt):",
        ctx.tenk.business[:8_000] or "(not extracted)",
        "",
        "Computed financial scale for reference:",
        ctx.financials_brief(),
    ]
    if feedback:
        parts += ["", "Reviewer feedback on your previous attempt — fix these:", feedback]
    return "\n".join(parts)


async def run(ctx: RunContext, cwd: Path, on_event: EventCallback | None = None, feedback: str | None = None) -> AgentRun:
    return await run_structured_agent(
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_prompt(ctx, feedback),
        output_model=PeerSelectionOutput,
        max_turns=16,
        cwd=cwd,
        on_event=on_event,
    )
