"""Runs the full pipeline for one ticker: deterministic data -> parallel research subagents ->
peer metrics -> thesis -> compose -> QA (with one targeted regeneration pass) -> final HTML."""
import asyncio
import json
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import compositor, edgar_client, financial_math, price_client
from .agents import (
    catalysts_risks,
    company_overview,
    financials_enrichment,
    news_digest,
    peer_selection,
    qa_review,
    thesis,
)
from .agents.base import AgentRun, to_markdown
from .config import OUTPUT_DIR
from .context import RunContext

RESEARCH_AGENTS = [company_overview, financials_enrichment, peer_selection, catalysts_risks, news_digest]
AGENT_TITLES = {m.NAME: m.TITLE for m in RESEARCH_AGENTS + [thesis, qa_review]}
STEP_ORDER = [
    "data_fetch",
    company_overview.NAME,
    financials_enrichment.NAME,
    peer_selection.NAME,
    catalysts_risks.NAME,
    news_digest.NAME,
    "peer_metrics",
    thesis.NAME,
    "compose",
    qa_review.NAME,
]
STEP_TITLES = {
    "data_fetch": "SEC EDGAR + market data (deterministic)",
    "peer_metrics": "Peer multiples (deterministic)",
    "compose": "Compose one-pager",
    **AGENT_TITLES,
}

# Free-text QA `section` values -> owning research agent
_SECTION_OWNERS = {
    "overview": company_overview.NAME,
    "business": company_overview.NAME,
    "snapshot": company_overview.NAME,
    "saas": financials_enrichment.NAME,
    "financial": financials_enrichment.NAME,
    "peer": peer_selection.NAME,
    "valuation": peer_selection.NAME,
    "catalyst": catalysts_risks.NAME,
    "risk": catalysts_risks.NAME,
    "news": news_digest.NAME,
    "guidance": news_digest.NAME,
    "thesis": thesis.NAME,
    "bull": thesis.NAME,
    "bear": thesis.NAME,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StepState:
    name: str
    title: str
    status: str = "pending"  # pending | running | done | failed | skipped
    detail: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    cost_usd: float = 0.0
    turns: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunState:
    run_id: str
    ticker: str
    focus_note: str
    status: str = "running"  # running | completed | failed
    partial: bool = False
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    company_name: str = ""
    steps: dict[str, StepState] = field(default_factory=dict)
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    final_html: str | None = None
    qa: dict[str, Any] | None = None
    error: str | None = None
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(self.steps[k]) for k in STEP_ORDER if k in self.steps]
        return d


class Pipeline:
    def __init__(self, ticker: str, focus_note: str = "", run_id: str | None = None):
        self.state = RunState(run_id=run_id or uuid.uuid4().hex[:12], ticker=ticker.upper().strip(), focus_note=focus_note.strip())
        for name in STEP_ORDER:
            self.state.steps[name] = StepState(name=name, title=STEP_TITLES[name])
        self.run_dir = OUTPUT_DIR / self.state.run_id
        (self.run_dir / "sections").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        (self.run_dir / "agent_cwd").mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self.sections: dict[str, Any] = {}
        self.agent_runs: dict[str, AgentRun] = {}
        self.ctx: RunContext | None = None
        self._persist()

    # ---------- status plumbing ----------
    def _persist(self) -> None:
        self.state.total_cost_usd = round(sum(s.cost_usd for s in self.state.steps.values()), 4)
        (self.run_dir / "status.json").write_text(json.dumps(self.state.to_dict(), indent=1, default=str), encoding="utf-8")

    def _step(self, name: str, status: str, detail: str = "") -> None:
        s = self.state.steps[name]
        s.status = status
        if detail:
            s.detail = detail
        if status == "running":
            s.started_at = _now()
        if status in ("done", "failed", "skipped"):
            s.finished_at = _now()
        self._persist()

    async def on_event(self, name: str, kind: str, detail: dict[str, Any]) -> None:
        async with self._lock:
            s = self.state.steps[name]
            if kind == "started":
                s.status = "running"
                s.started_at = _now()
            elif kind == "tool":
                s.detail = f"{detail['tool']}: {detail['input'][:120]}"
                s.events.append({"t": _now(), **detail})
                s.events = s.events[-40:]
            elif kind in ("finished", "failed"):
                s.cost_usd = round(s.cost_usd + float(detail.get("cost_usd") or 0), 4)
                s.turns += int(detail.get("turns") or 0)
                s.detail = "" if kind == "finished" else f"Failed: {detail.get('error')}"
            self._persist()

    def _save_section(self, name: str, title: str, run: AgentRun) -> None:
        base = self.run_dir / "sections" / name
        payload = {
            "agent": name,
            "title": title,
            "ok": run.ok,
            "error": run.error,
            "cost_usd": round(run.cost_usd, 4),
            "turns": run.num_turns,
            "tool_calls": run.tool_calls,
            "data": run.data,
        }
        base.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        md = to_markdown(title, run.data) if run.data else f"# {title}\n\nUnavailable: {run.error}\n"
        base.with_suffix(".md").write_text(md, encoding="utf-8")
        self.state.outputs[name] = {"json": f"sections/{name}.json", "md": f"sections/{name}.md", "title": title}
        self.agent_runs[name] = run
        if name != qa_review.NAME:  # the reviewer's verdict is not page content
            self.sections[name] = run.data

    # ---------- pipeline ----------
    async def run(self) -> RunState:
        try:
            async with httpx.AsyncClient(follow_redirects=True) as http:
                await self._data_fetch(http)
                await self._research_agents()
                await self._peer_metrics(http)
                await self._thesis()
                self._compose()
                await self._qa_and_repair(http)
            self.state.status = "completed"
            self.state.partial = any(s.status == "failed" for s in self.state.steps.values())
        except Exception as exc:
            self.state.status = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
            for s in self.state.steps.values():
                if s.status == "running":
                    s.status = "failed"
                    s.detail = self.state.error
        self.state.finished_at = _now()
        self._persist()
        return self.state

    async def _data_fetch(self, http: httpx.AsyncClient) -> None:
        self._step("data_fetch", "running", "Resolving ticker on SEC EDGAR")
        company = await edgar_client.resolve_company(http, self.state.ticker)
        self.state.company_name = company.name
        tenk_filing = company.latest("10-K", "10-K/A")
        if tenk_filing is None:
            raise RuntimeError(f"No 10-K found on EDGAR for {self.state.ticker}; only US domestic filers are supported")
        self._step("data_fetch", "running", f"Fetching XBRL facts, prices and the {tenk_filing.report_date} 10-K")
        facts, quote, tenk = await asyncio.gather(
            edgar_client.get_company_facts(http, company.cik),
            price_client.get_quote(http, self.state.ticker),
            edgar_client.get_10k_sections(http, tenk_filing),
        )
        fin = financial_math.compute_financials(facts)
        val = financial_math.compute_valuation(fin, quote.price, quote.monthly_closes)
        last_8k = edgar_client.latest_earnings_8k(company)
        self.ctx = RunContext(
            ticker=self.state.ticker,
            focus_note=self.state.focus_note,
            company=company,
            quote=quote,
            financials=fin,
            valuation=val,
            tenk=tenk,
            last_8k_date=last_8k.filing_date if last_8k else None,
        )
        data = {
            "company": {k: v for k, v in asdict(company).items() if k != "filings"},
            "tenk_url": tenk.url,
            "quote": {k: v for k, v in asdict(quote).items() if k != "monthly_closes"},
            "monthly_closes": quote.monthly_closes,
            "financials": asdict(fin),
            "valuation": asdict(val),
        }
        (self.run_dir / "data" / "market_and_financials.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        (self.run_dir / "data" / "financials_brief.md").write_text(f"# Computed financials — {company.name}\n\n```\n{self.ctx.financials_brief()}\n```\n", encoding="utf-8")
        self.state.outputs["data_fetch"] = {"json": "data/market_and_financials.json", "md": "data/financials_brief.md", "title": STEP_TITLES["data_fetch"]}
        self.sections["financials"] = data["financials"]
        self.sections["valuation"] = data["valuation"]
        self._step("data_fetch", "done", f"{company.name} · {len(fin.periods)} fiscal years + TTM · price ${quote.price:.2f}")

    async def _run_agent(self, module, feedback: str | None = None) -> AgentRun:
        assert self.ctx is not None
        run = await module.run(self.ctx, cwd=self.run_dir / "agent_cwd", on_event=self.on_event, feedback=feedback)
        self._save_section(module.NAME, module.TITLE, run)
        self._step(module.NAME, "done" if run.ok else "failed", "" if run.ok else f"Failed: {run.error}")
        return run

    async def _research_agents(self) -> None:
        await asyncio.gather(*(self._run_agent(m) for m in RESEARCH_AGENTS))

    async def _peer_metrics(self, http: httpx.AsyncClient) -> None:
        assert self.ctx is not None
        self._step("peer_metrics", "running")
        peers = (self.sections.get(peer_selection.NAME) or {}).get("peers") or []
        if not peers:
            self.sections["peer_metrics"] = {"peers": [], "notes": ["Peer selection unavailable"]}
            self._step("peer_metrics", "skipped", "No peers to compute")
            return
        results = await asyncio.gather(*(self._one_peer(http, p) for p in peers[:5]), return_exceptions=True)
        rows, notes = [], []
        for p, r in zip(peers, results):
            if isinstance(r, Exception):
                notes.append(f"{p['ticker']}: {type(r).__name__}: {str(r)[:120]}")
            else:
                rows.append(r)
        target = self._target_row()
        medians = {}
        for k in ("ev_to_revenue", "ev_to_ebitda", "price_to_fcf", "price_to_earnings", "revenue_growth_pct"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            medians[k] = round(statistics.median(vals), 1) if vals else None
        premium = None
        if target["ev_to_revenue"] and medians.get("ev_to_revenue"):
            premium = round(100 * (target["ev_to_revenue"] / medians["ev_to_revenue"] - 1), 1)
        self.sections["peer_metrics"] = {
            "target": target,
            "peers": rows,
            "peer_median": medians,
            "ev_rev_premium_vs_median_pct": premium,
            "notes": notes,
            "as_of": self.ctx.quote.as_of,
        }
        (self.run_dir / "data" / "peer_metrics.json").write_text(json.dumps(self.sections["peer_metrics"], indent=1), encoding="utf-8")
        self._step("peer_metrics", "done" if rows else "failed", f"{len(rows)}/{len(peers)} peers computed" + (f"; issues: {len(notes)}" if notes else ""))

    def _target_row(self) -> dict[str, Any]:
        assert self.ctx is not None
        v, t = self.ctx.valuation, self.ctx.financials.ttm
        return {
            "ticker": self.ctx.ticker,
            "name": self.ctx.company.name,
            "market_cap": v.market_cap,
            "ev_to_revenue": v.ev_to_revenue,
            "ev_to_ebitda": v.ev_to_ebitda,
            "price_to_fcf": v.price_to_fcf,
            "price_to_earnings": v.price_to_earnings,
            "revenue_growth_pct": t.revenue_growth_pct if t else None,
            "fcf_margin_pct": t.fcf_margin_pct if t else None,
        }

    async def _one_peer(self, http: httpx.AsyncClient, peer: dict[str, Any]) -> dict[str, Any]:
        ticker = peer["ticker"].upper()
        company = await edgar_client.resolve_company(http, ticker)
        facts, quote = await asyncio.gather(edgar_client.get_company_facts(http, company.cik), price_client.get_quote(http, ticker, years=1))
        fin = financial_math.compute_financials(facts)
        val = financial_math.compute_valuation(fin, quote.price, [])
        t = fin.ttm
        return {
            "ticker": ticker,
            "name": company.name,
            "rationale": peer.get("rationale", ""),
            "market_cap": val.market_cap,
            "ev_to_revenue": val.ev_to_revenue,
            "ev_to_ebitda": val.ev_to_ebitda,
            "price_to_fcf": val.price_to_fcf,
            "price_to_earnings": val.price_to_earnings,
            "revenue_growth_pct": t.revenue_growth_pct if t else None,
            "fcf_margin_pct": t.fcf_margin_pct if t else None,
        }

    async def _thesis(self, feedback: str | None = None) -> None:
        assert self.ctx is not None
        run = await thesis.run(self.ctx, self.sections, cwd=self.run_dir / "agent_cwd", on_event=self.on_event, feedback=feedback)
        self._save_section(thesis.NAME, thesis.TITLE, run)
        self._step(thesis.NAME, "done" if run.ok else "failed", "" if run.ok else f"Failed: {run.error}")

    def _compose(self) -> None:
        assert self.ctx is not None
        self._step("compose", "running")
        html, problems, over_budget = compositor.compose(self.ctx, self.sections, self.state.to_dict())
        out = self.run_dir / "onepager.html"
        out.write_text(html, encoding="utf-8")
        self.state.final_html = "onepager.html"
        (self.run_dir / "assembled.json").write_text(json.dumps(self.sections, indent=1, default=str), encoding="utf-8")
        detail = "; ".join(problems) if problems else "Structural check passed"
        if over_budget:
            detail += f" · {len(over_budget)} field(s) over length budget, truncated on page"
        self._step("compose", "done" if not problems else "failed", detail)

    async def _qa_and_repair(self, http: httpx.AsyncClient) -> None:
        assert self.ctx is not None
        qa = await qa_review.run(self.ctx, self.sections, cwd=self.run_dir / "agent_cwd", on_event=self.on_event)
        self._save_section(qa_review.NAME, qa_review.TITLE, qa)
        self.state.qa = qa.data
        if not qa.ok:
            self._step(qa_review.NAME, "failed", f"QA agent failed: {qa.error}")
            return
        actionable = [i for i in qa.data["issues"] if i["severity"] in ("blocker", "major")]
        if not actionable:
            self._step(qa_review.NAME, "done", f"Passed · {len(qa.data['issues'])} minor note(s)")
            return
        self._step(qa_review.NAME, "running", f"{len(actionable)} issue(s) — regenerating affected sections")
        feedback: dict[str, list[str]] = {}
        for issue in actionable:
            owner = next((v for k, v in _SECTION_OWNERS.items() if k in issue["section"].lower()), None)
            if owner:
                feedback.setdefault(owner, []).append(f"[{issue['severity']}] {issue['description']} Fix: {issue['suggested_fix']}")
        research_fixes = [m for m in RESEARCH_AGENTS if m.NAME in feedback]
        if research_fixes:
            await asyncio.gather(*(self._run_agent(m, feedback="\n".join(feedback[m.NAME])) for m in research_fixes))
            if peer_selection.NAME in feedback:
                await self._peer_metrics(http)
        thesis_feedback = "\n".join(feedback.get(thesis.NAME, []))
        if research_fixes or thesis_feedback:
            await self._thesis(feedback=thesis_feedback or "Upstream sections were regenerated; re-synthesize from the updated inputs.")
        self._compose()
        qa2 = await qa_review.run(self.ctx, self.sections, cwd=self.run_dir / "agent_cwd", on_event=self.on_event)
        self._save_section(qa_review.NAME, qa_review.TITLE, qa2)
        self.state.qa = qa2.data or qa.data
        if qa2.ok:
            remaining = [i for i in qa2.data["issues"] if i["severity"] == "blocker"]
            self._step(qa_review.NAME, "done", "Passed after one repair pass" if not remaining else f"{len(remaining)} blocker(s) remain — review manually")
        else:
            self._step(qa_review.NAME, "failed", f"Second QA pass failed: {qa2.error}")


def load_state(run_id: str) -> dict[str, Any] | None:
    p = OUTPUT_DIR / run_id / "status.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_runs() -> list[dict[str, Any]]:
    runs = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            st = load_state(d.name)
            if st:
                runs.append({k: st.get(k) for k in ("run_id", "ticker", "company_name", "status", "partial", "created_at", "total_cost_usd")})
    return sorted(runs, key=lambda r: r["created_at"] or "", reverse=True)


def run_dir(run_id: str) -> Path:
    return OUTPUT_DIR / run_id
