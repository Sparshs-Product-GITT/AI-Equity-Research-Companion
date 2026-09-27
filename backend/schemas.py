"""Output contracts for every LLM subagent. Passed to the Agent SDK as structured-output JSON
Schema (which validates and re-prompts on violation) and validated again on the way back.

Character and item limits are the one-pager's content budget: they are enforced, not advisory."""
from enum import Enum

from pydantic import BaseModel, Field

NO_URL = " Plain analyst prose: no URLs, no source names, no parentheticals longer than a clause."


class Source(BaseModel):
    title: str = Field(max_length=60, description="Short label, e.g. 'FY2026 10-K, Item 1'")
    url: str = Field(description="Direct URL to the document or page")


class Segment(BaseModel):
    name: str = Field(max_length=45)
    pct_of_revenue: float | None = Field(default=None, description="Share of total revenue in percent, if disclosed")
    note: str | None = Field(default=None, max_length=60, description="Optional, e.g. '+23% y/y incl. Informatica'")


class GeoMix(BaseModel):
    region: str = Field(max_length=30)
    pct_of_revenue: float | None = None


class CompanyOverviewOutput(BaseModel):
    one_liner: str = Field(max_length=160, description="One sentence, <= 25 words, what the company does")
    business_model: str = Field(max_length=220, description="How it earns revenue and the recurring share, one or two sentences." + NO_URL)
    description: str = Field(max_length=380, description="2-3 factual sentences from the 10-K on what it sells and to whom." + NO_URL)
    segments: list[Segment] = Field(max_length=6, description="Reportable segments or product lines with revenue share")
    geographic_mix: list[GeoMix] = Field(default_factory=list, max_length=4)
    customer_base: str = Field(max_length=200, description="Who buys and how it is sold, one or two sentences." + NO_URL)
    key_products: list[str] = Field(max_length=6, description="3-6 product names, <= 40 chars each")
    sub_industry: str = Field(max_length=50, description="<= 6 words, e.g. 'Enterprise application software (CRM)'")
    sources: list[Source] = Field(max_length=5)


class SaaSMetric(BaseModel):
    name: str = Field(max_length=40, description="e.g. 'Current RPO', 'Net revenue retention'")
    value: str = Field(max_length=25, description="As disclosed, with unit, e.g. '$31.2B (+16% y/y)'")
    period: str = Field(max_length=25, description="e.g. 'Q2 FY27' or '2026-07-31'")
    source: Source


class FinancialsEnrichmentOutput(BaseModel):
    saas_metrics: list[SaaSMetric] = Field(max_length=5, description="Only metrics the company itself discloses; empty list if none")
    reporting_notes: list[str] = Field(default_factory=list, max_length=3, description="<= 140 chars each: fiscal-year convention, one-offs, restatements a reader must know." + NO_URL)
    sources: list[Source] = Field(max_length=5)


class Peer(BaseModel):
    ticker: str = Field(max_length=6, description="US ticker symbol")
    name: str = Field(max_length=40)
    rationale: str = Field(max_length=90, description="One clause on why it is a direct comparable")


class PeerSelectionOutput(BaseModel):
    peers: list[Peer] = Field(min_length=3, max_length=5, description="3 to 5 direct, US-listed public comparables")
    selection_rationale: str = Field(max_length=200, description="One sentence on the screen used (sub-industry, model, scale)." + NO_URL)
    sources: list[Source] = Field(max_length=5)


class Catalyst(BaseModel):
    event: str = Field(max_length=60)
    expected_date: str | None = Field(default=None, max_length=20, description="ISO date or approximate, e.g. '2026-11-25' or 'Q4 2026'")
    why_it_matters: str = Field(max_length=110, description="One clause." + NO_URL)
    source: Source


class RiskFactor(BaseModel):
    title: str = Field(max_length=45, description="<= 6 words")
    summary: str = Field(max_length=160, description="One sentence paraphrasing the company's own disclosure." + NO_URL)
    source: Source


class CatalystsRisksOutput(BaseModel):
    next_earnings_date: str | None = Field(default=None, max_length=20, description="Confirmed or estimated next earnings date, ISO if known")
    catalysts: list[Catalyst] = Field(min_length=2, max_length=4)
    risks: list[RiskFactor] = Field(min_length=3, max_length=4, description="Most material, company-specific 10-K Item 1A risks; no boilerplate")
    sources: list[Source] = Field(max_length=5)


class GuidanceDirection(str, Enum):
    raised = "raised"
    lowered = "lowered"
    maintained = "maintained"
    initiated = "initiated"
    not_provided = "not_provided"
    unknown = "unknown"


class NewsCategory(str, Enum):
    earnings = "earnings"
    product = "product"
    m_and_a = "m_and_a"
    leadership = "leadership"
    regulatory = "regulatory"
    other = "other"


class NewsItem(BaseModel):
    headline: str = Field(max_length=90)
    date: str = Field(max_length=10, description="ISO date")
    category: NewsCategory
    summary: str = Field(max_length=130, description="One sentence: what happened and why it matters." + NO_URL)
    url: str


class NewsDigestOutput(BaseModel):
    last_earnings_date: str | None = Field(default=None, max_length=10, description="Most recent quarterly earnings release date, ISO")
    guidance_direction: GuidanceDirection
    guidance_note: str = Field(max_length=160, description="Metric, prior vs new, in one sentence; or why unknown." + NO_URL)
    items: list[NewsItem] = Field(min_length=1, max_length=5, description="Most material company-specific items since last earnings, newest first")
    sources: list[Source] = Field(max_length=5)


class ThesisOutput(BaseModel):
    key_debate: str = Field(max_length=200, description="One sentence naming the central tension. No rating, target price, sources or URLs.")
    bull_case: list[str] = Field(min_length=3, max_length=4, description="<= 150 chars each, each anchored to a specific number from the inputs. No sources or URLs.")
    bear_case: list[str] = Field(min_length=3, max_length=4, description="<= 150 chars each, each anchored to a specific number from the inputs. No sources or URLs.")


class QAIssue(BaseModel):
    section: str = Field(max_length=60)
    severity: str = Field(max_length=10, description="'blocker' | 'major' | 'minor'")
    description: str = Field(max_length=400)
    suggested_fix: str = Field(max_length=400)


class QAReviewOutput(BaseModel):
    passed: bool = Field(description="False if any blocker exists")
    issues: list[QAIssue] = Field(max_length=10)
    summary: str = Field(max_length=300)


def schema_for(model: type[BaseModel]) -> dict:
    return model.model_json_schema()
