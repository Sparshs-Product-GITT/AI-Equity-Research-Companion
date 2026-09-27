from pathlib import Path

from ..context import RunContext
from ..schemas import CatalystsRisksOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "catalysts_risks"
TITLE = "Catalysts & risks"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: Catalysts & Risks. Facts and dated events, not opinions.

Catalysts (2-4): near-term, datable events that could move the stock — the next earnings release date (confirm on the IR site or from the company's typical cadence and say which), scheduled investor/analyst days, major product launches or GA dates announced by the company, pending acquisitions awaiting close or regulatory approval, contract renewals or legal decisions with known timing. Each needs a source URL.

Risks (3-4): the most material, company-specific risk factors from the 10-K Item 1A excerpt provided. Skip generic boilerplate ('our stock price may be volatile', 'we face competition', 'macroeconomic conditions', 'we may be subject to litigation'). Prefer risks that are specific to this company's strategy, customer concentration, pricing model, technology transition, regulatory exposure, or capital structure. Paraphrase the company's own language in 1-2 sentences; cite the 10-K URL. If a recent event has sharpened a risk (e.g. a new regulation, a lost customer), you may add one sourced current-events risk.

`next_earnings_date`: ISO date if confirmed; if only estimated, give the estimate and say 'estimated' in the catalyst text."""


def build_prompt(ctx: RunContext, feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        f"Latest 10-K: {ctx.tenk.url}",
        f"Most recent 8-K filing date (often the last earnings release): {ctx.last_8k_date or 'unknown'}",
        "",
        "=== 10-K Item 1A (Risk Factors) excerpt ===",
        ctx.tenk.risk_factors or "(section not extracted; fetch the 10-K yourself)",
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
        output_model=CatalystsRisksOutput,
        max_turns=20,
        cwd=cwd,
        on_event=on_event,
    )
