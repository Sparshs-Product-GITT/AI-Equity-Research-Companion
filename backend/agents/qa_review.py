import json
from pathlib import Path
from typing import Any

from ..context import RunContext
from ..schemas import QAReviewOutput
from .base import AgentRun, EventCallback, run_structured_agent

NAME = "qa_review"
TITLE = "QA review"

SYSTEM_PROMPT = """You are the reviewing editor for a one-page equity research pre-read assembled from several independent research subagents plus deterministic financial computations. You have no research tools. Your job is to catch what would embarrass a senior analyst before the page ships.

Check, section by section:
1. Cross-section consistency: every number quoted in the thesis bullets or news summaries must match the computed financials/valuation given (same figure, same period). Flag any mismatch with both values.
2. Grounding: each bull/bear bullet must trace to a fact present in the inputs. Flag bullets that introduce new claims.
3. Guidance badge vs. news: `guidance_direction` must be consistent with the earnings item in `news_digest.items` and `guidance_note`.
4. Windows and dates: news items must be dated on/after `last_earnings_date` (or within 90 days if unknown); catalysts must be forward-looking relative to today; ISO dates well-formed.
5. Risk quality: risks must be company-specific, not boilerplate; 3-4 of them.
6. Peers: 3-5 peers, plausibly same sub-industry; flag any that are obviously not comparable or not US filers.
7. Sources: every section has at least one source URL; URLs are plausible (sec.gov, IR domains, major outlets).
8. Tone: no ratings, price targets, or 'we recommend' language anywhere; one-liner <= 25 words; sentence case.
9. Brevity: this is a one-pager. Flag (minor) any narrative field that reads as a paragraph, repeats a number shown in a table, or embeds URLs/source names in prose.

Severity: `blocker` = factual contradiction, fabricated claim, or rating/target language; `major` = missing required content or wrong window; `minor` = style. `passed` is false only if a blocker exists. Be specific in `suggested_fix` so the owning subagent can act on it without re-reading everything."""


def build_prompt(ctx: RunContext, assembled: dict[str, Any]) -> str:
    return "\n".join(
        [
            ctx.header,
            "",
            "=== Computed financials & valuation (ground truth) ===",
            ctx.financials_brief(),
            "",
            "=== Assembled one-pager data (all sections) ===",
            json.dumps(assembled, indent=1, default=str),
        ]
    )


async def run(ctx: RunContext, assembled: dict[str, Any], cwd: Path, on_event: EventCallback | None = None) -> AgentRun:
    return await run_structured_agent(
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_prompt(ctx, assembled),
        output_model=QAReviewOutput,
        tools=(),
        max_turns=4,
        cwd=cwd,
        on_event=on_event,
    )
