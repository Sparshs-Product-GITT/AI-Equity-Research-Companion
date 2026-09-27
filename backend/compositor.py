"""Deterministic HTML assembly: a fixed two-column template populated from computed data and
subagent JSON. Only narrative text comes from the LLM; layout never does."""
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import TEMPLATE_DIR
from .context import RunContext, fmt_pct, fmt_usd, fmt_x

_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
_env.filters.update({"usd": fmt_usd, "pct": fmt_pct, "x": fmt_x})

REQUIRED_IDS = ["hdr", "overview", "financials", "valuation", "thesis", "catalysts", "news", "footer"]

CATEGORY_LABELS = {
    "earnings": "Earnings",
    "product": "Product",
    "m_and_a": "M&A",
    "leadership": "Leadership",
    "regulatory": "Regulatory",
    "other": "Other",
}
GUIDANCE_LABELS = {
    "raised": ("Guidance: raised", "good"),
    "lowered": ("Guidance: lowered", "bad"),
    "maintained": ("Guidance: maintained", "neutral"),
    "initiated": ("Guidance: initiated", "neutral"),
    "not_provided": ("No guidance given", "muted"),
    "unknown": ("Guidance: not found", "muted"),
}


def _chart(periods: list[dict[str, Any]]) -> dict[str, Any]:
    bars = [p for p in periods if p.get("revenue")]
    if not bars:
        return {"bars": []}
    mx = max(p["revenue"] for p in bars)
    return {
        "bars": [
            {
                "label": p["label"],
                "height": round(100 * p["revenue"] / mx),
                "value": fmt_usd(p["revenue"], 1),
                "growth": fmt_pct(p.get("revenue_growth_pct")) if p.get("revenue_growth_pct") is not None else "",
            }
            for p in bars
        ]
    }


def _sources(section: dict[str, Any] | None, limit: int = 4) -> list[dict[str, str]]:
    if not section:
        return []
    seen, out = set(), []
    for s in section.get("sources") or []:
        if s.get("url") and s["url"] not in seen:
            seen.add(s["url"])
            out.append(s)
    return out[:limit]


def build_page(ctx: RunContext, sections: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    fin, val, q, c = ctx.financials, ctx.valuation, ctx.quote, ctx.company
    ov = sections.get("company_overview") or {}
    enrich = sections.get("financials_enrichment") or {}
    pm = sections.get("peer_metrics") or {}
    cr = sections.get("catalysts_risks") or {}
    nd = sections.get("news_digest") or {}
    th = sections.get("thesis_synthesis") or {}
    periods = [vars(p) if not isinstance(p, dict) else p for p in fin.periods] + ([vars(fin.ttm) if not isinstance(fin.ttm, dict) else fin.ttm] if fin.ttm else [])
    guidance_label, guidance_tone = GUIDANCE_LABELS.get(nd.get("guidance_direction", "unknown"), GUIDANCE_LABELS["unknown"])
    hist_note = None
    if val.ev_to_revenue and val.ev_to_revenue_3y_avg:
        rel = "below" if val.ev_to_revenue < val.ev_to_revenue_3y_avg else "above"
        hist_note = f"{ctx.ticker} trades at {fmt_x(val.ev_to_revenue)} trailing EV/Revenue, {rel} its 3-year average of {fmt_x(val.ev_to_revenue_3y_avg)} (range {fmt_x(val.ev_to_revenue_3y_low)}–{fmt_x(val.ev_to_revenue_3y_high)})."
    prem = pm.get("ev_rev_premium_vs_median_pct")
    prem_note = None
    if prem is not None:
        prem_note = f"{abs(prem):.0f}% {'premium' if prem > 0 else 'discount'} to the peer-median EV/Revenue of {fmt_x(pm['peer_median'].get('ev_to_revenue'))}."
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_id": state.get("run_id"),
        "ticker": ctx.ticker,
        "name": c.name,
        "exchange": q.exchange or ", ".join(c.exchanges or []),
        "sub_industry": ov.get("sub_industry") or c.sic_description,
        "one_liner": ov.get("one_liner") or "",
        "price": q.price,
        "as_of": q.as_of,
        "market_cap": val.market_cap,
        "wk52_low": q.fifty_two_week_low,
        "wk52_high": q.fifty_two_week_high,
        "shares": fin.balance.shares_outstanding,
        "fye": c.fiscal_year_end,
        "focus_note": ctx.focus_note,
        "kpis": _kpis(fin),
        "overview": ov,
        "overview_sources": _sources(ov),
        "periods": periods,
        "chart": _chart(periods),
        "balance": vars(fin.balance),
        "fin_notes": fin.notes + val.notes,
        "saas_metrics": enrich.get("saas_metrics") or [],
        "reporting_notes": enrich.get("reporting_notes") or [],
        "fin_sources": _sources(enrich) + [{"title": "SEC XBRL company facts", "url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c.cik:010d}.json"}],
        "tenk_url": ctx.tenk.url,
        "valuation": vars(val),
        "hist_note": hist_note,
        "peer_metrics": pm,
        "prem_note": prem_note,
        "peer_rationale": (sections.get("peer_selection") or {}).get("selection_rationale"),
        "peer_sources": _sources(sections.get("peer_selection")),
        "thesis": th,
        "catalysts": cr.get("catalysts") or [],
        "risks": cr.get("risks") or [],
        "next_earnings": cr.get("next_earnings_date"),
        "cr_sources": _sources(cr),
        "guidance_label": guidance_label,
        "guidance_tone": guidance_tone,
        "guidance_note": nd.get("guidance_note"),
        "last_earnings_date": nd.get("last_earnings_date"),
        "news": [{**i, "category_label": CATEGORY_LABELS.get(i.get("category", "other"), "Other")} for i in (nd.get("items") or [])],
        "news_sources": _sources(nd),
        "unavailable": {k: (sections.get(k) is None) for k in ("company_overview", "financials_enrichment", "peer_selection", "catalysts_risks", "news_digest", "thesis_synthesis")},
        "qa": state.get("qa"),
        "partial": state.get("partial"),
    }


def _kpis(fin) -> list[dict[str, str]]:
    m = fin.ttm or (fin.periods[-1] if fin.periods else None)
    if m is None:
        return []
    label = m.label if m.label == "TTM" else m.label
    return [
        {"label": f"Revenue growth ({label})", "value": fmt_pct(m.revenue_growth_pct)},
        {"label": f"Gross margin ({label})", "value": fmt_pct(m.gross_margin_pct)},
        {"label": f"Operating margin ({label})", "value": fmt_pct(m.operating_margin_pct)},
        {"label": f"FCF margin ({label})", "value": fmt_pct(m.fcf_margin_pct)},
        {"label": "Rule of 40", "value": "n/a" if m.rule_of_40 is None else f"{m.rule_of_40:.0f}"},
    ]


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.depth = 0
        self.max_depth = 0

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k == "id" and v:
                self.ids.add(v)
        if tag not in ("br", "img", "meta", "link", "input", "hr"):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)

    def handle_endtag(self, tag):
        if tag not in ("br", "img", "meta", "link", "input", "hr"):
            self.depth -= 1


def structural_check(html: str) -> list[str]:
    problems = []
    p = _IdCollector()
    p.feed(html)
    missing = [i for i in REQUIRED_IDS if i not in p.ids]
    if missing:
        problems.append(f"missing sections: {', '.join(missing)}")
    if p.depth != 0:
        problems.append(f"unbalanced tags (depth {p.depth} at end)")
    for token in ("{{", "{%", "None</", ">None<"):
        if token in html:
            problems.append(f"unrendered token {token!r} present")
    if re.search(r">\s*(nan|NaN|inf)\s*<", html):
        problems.append("non-finite number rendered")
    if len(html) < 8_000:
        problems.append("page suspiciously small")
    return problems


NARRATIVE_BUDGET = 400  # chars; anything longer is a paragraph, not a one-pager note


def length_audit(sections: dict[str, Any]) -> list[str]:
    """Narrative fields over budget. The template truncates them anyway; this makes it visible."""
    over: list[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str) and "url" not in path.lower() and len(o) > NARRATIVE_BUDGET:
            over.append(f"{path} ({len(o)} chars)")

    for key in ("company_overview", "financials_enrichment", "peer_selection", "catalysts_risks", "news_digest", "thesis_synthesis"):
        if sections.get(key):
            walk(sections[key], key)
    return over


def compose(ctx: RunContext, sections: dict[str, Any], state: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    page = build_page(ctx, sections, state)
    html = _env.get_template("onepager.html").render(page=page)
    return html, structural_check(html), length_audit(sections)
