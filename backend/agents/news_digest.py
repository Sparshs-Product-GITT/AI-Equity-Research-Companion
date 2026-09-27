from pathlib import Path

from ..context import RunContext
from ..schemas import NewsDigestOutput
from .base import ANALYST_PREAMBLE, AgentRun, EventCallback, run_structured_agent

NAME = "news_digest"
TITLE = "Recent news & events (since last earnings)"

SYSTEM_PROMPT = ANALYST_PREAMBLE + """

Your section: Recent News & Events, windowed from the most recent quarterly earnings release to today.

Method:
1. Establish `last_earnings_date`: find the most recent quarterly results press release on the company's IR site or newsroom (the most recent 8-K filing date is given as a hint — confirm it). If you cannot confirm it, use a 90-day window and say so in `guidance_note`.
2. Guidance: from that earnings release or call, determine whether the company raised, lowered, maintained or initiated its forward guidance (revenue and/or EPS/margin), and state the specific change in `guidance_note` (metric, prior vs new). If the company does not give guidance, use `not_provided`. If you could not find it, `unknown`.
3. Items: the 3-6 most material company-specific developments in the window, newest first — the earnings result itself (beat/miss vs company guidance, not vs unnamed 'estimates' unless a source states them), acquisitions, executive changes, major product/AI launches, pricing changes, regulatory or legal actions, large customer wins/losses, buybacks or debt issuance. Skip stock-price-move stories, analyst rating changes, and generic market commentary.
4. Every item needs an ISO date and a URL. Prefer the company's press release; otherwise a major outlet (Reuters, Bloomberg, WSJ, CNBC, The Information)."""


def build_prompt(ctx: RunContext, feedback: str | None = None) -> str:
    parts = [
        ctx.header,
        "",
        f"Hint — most recent 8-K filing date: {ctx.last_8k_date or 'unknown'}",
        f"Latest 10-K: {ctx.tenk.url}",
    ]
    if feedback:
        parts += ["", "Reviewer feedback on your previous attempt — fix these:", feedback]
    return "\n".join(parts)


async def run(ctx: RunContext, cwd: Path, on_event: EventCallback | None = None, feedback: str | None = None) -> AgentRun:
    return await run_structured_agent(
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_prompt(ctx, feedback),
        output_model=NewsDigestOutput,
        max_turns=24,
        cwd=cwd,
        on_event=on_event,
    )
