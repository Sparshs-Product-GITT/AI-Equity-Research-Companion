"""Deterministic financial computations from SEC XBRL company facts + market prices.

All heavy numbers on the one-pager come from here, never from an LLM.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Any

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
DURATION_TAGS = {
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "snm": ["SellingAndMarketingExpense", "SellingGeneralAndAdministrativeExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfServices"],
    "dna": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndOther",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
}
INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"],
    "debt_current": ["LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
}
SHARES_TAGS = ["EntityCommonStockSharesOutstanding"]


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _facts(company_facts: dict[str, Any], taxonomy: str, tag: str) -> list[dict[str, Any]]:
    units = company_facts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {})
    for unit in ("USD", "USD/shares", "shares", "pure"):
        if unit in units:
            return units[unit]
    return next(iter(units.values()), [])


def _first_tag_with_data(company_facts: dict[str, Any], tags: list[str], taxonomy: str = "us-gaap") -> tuple[str, list[dict[str, Any]]]:
    for tag in tags:
        rows = _facts(company_facts, taxonomy, tag)
        if rows:
            return tag, rows
    return "", []


def _annual_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """fiscal-year-end date -> row, keeping the most recently filed value per period."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("fp") != "FY" or not r.get("form", "").startswith("10-K") or "start" not in r:
            continue
        days = (_d(r["end"]) - _d(r["start"])).days
        if not 340 <= days <= 380:
            continue
        prev = out.get(r["end"])
        if prev is None or r.get("filed", "") >= prev.get("filed", ""):
            out[r["end"]] = r
    return out


def _ytd_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """10-Q duration rows (quarterly or year-to-date), deduped to latest filing per (start,end)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if not r.get("form", "").startswith("10-Q") or "start" not in r:
            continue
        key = (r["start"], r["end"])
        prev = out.get(key)
        if prev is None or r.get("filed", "") >= prev.get("filed", ""):
            out[key] = r
    return sorted(out.values(), key=lambda r: (r["end"], r["start"]))


def _ttm(rows: list[dict[str, Any]], annual: dict[str, dict[str, Any]]) -> tuple[float | None, str, str]:
    """Trailing twelve months = latest FY + current YTD - prior-year YTD.

    Returns (value, period_end, method). Falls back to the latest FY when no 10-Q
    follows the last 10-K.
    """
    if not annual:
        return None, "", "unavailable"
    fy_end = max(annual)
    fy_val = float(annual[fy_end]["val"])
    ytd = [r for r in _ytd_rows(rows) if r["end"] > fy_end]
    if not ytd:
        return fy_val, fy_end, "fiscal_year"
    latest_end = max(r["end"] for r in ytd)
    cur = max((r for r in ytd if r["end"] == latest_end), key=lambda r: (_d(r["end"]) - _d(r["start"])).days)
    cur_days = (_d(cur["end"]) - _d(cur["start"])).days
    prior_candidates = [
        r
        for r in _ytd_rows(rows)
        if abs((_d(cur["end"]) - _d(r["end"])).days - 365) <= 10
        and abs((_d(r["end"]) - _d(r["start"])).days - cur_days) <= 10
    ]
    if not prior_candidates:
        return fy_val, fy_end, "fiscal_year"
    prior = prior_candidates[-1]
    return fy_val + float(cur["val"]) - float(prior["val"]), latest_end, "fy_plus_ytd_delta"


def _latest_instant(rows: list[dict[str, Any]]) -> tuple[float | None, str]:
    inst = [r for r in rows if "start" not in r and r.get("form", "").startswith(("10-K", "10-Q"))]
    if not inst:
        return None, ""
    r = max(inst, key=lambda r: (r["end"], r.get("filed", "")))
    return float(r["val"]), r["end"]


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return round(100.0 * num / den, 1)


def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or not prev:
        return None
    return round(100.0 * (cur - prev) / abs(prev), 1)


@dataclass
class PeriodMetrics:
    label: str
    period_end: str
    revenue: float | None = None
    revenue_growth_pct: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    eps_diluted: float | None = None
    rnd_pct_revenue: float | None = None
    snm_pct_revenue: float | None = None
    fcf: float | None = None
    fcf_margin_pct: float | None = None
    rule_of_40: float | None = None
    ebitda: float | None = None


@dataclass
class BalanceSnapshot:
    cash_and_investments: float | None
    total_debt: float | None
    net_cash: float | None
    as_of: str
    shares_outstanding: float | None
    shares_as_of: str


@dataclass
class Financials:
    revenue_tag: str
    snm_tag: str
    periods: list[PeriodMetrics]
    ttm: PeriodMetrics | None
    balance: BalanceSnapshot
    notes: list[str] = field(default_factory=list)


def compute_financials(company_facts: dict[str, Any], years: int = 3) -> Financials:
    notes: list[str] = []
    rev_tag, rev_rows = _first_tag_with_data(company_facts, REVENUE_TAGS)
    if not rev_rows:
        raise ValueError("No revenue facts found in XBRL company facts")
    series: dict[str, tuple[str, list[dict[str, Any]]]] = {"revenue": (rev_tag, rev_rows)}
    for key, tags in DURATION_TAGS.items():
        series[key] = _first_tag_with_data(company_facts, tags)
    if not series["gross_profit"][1] and series["cogs"][1]:
        notes.append("Gross profit derived as revenue minus cost of revenue (no GrossProfit tag).")
    if series["snm"][0] == "SellingGeneralAndAdministrativeExpense":
        notes.append("Company reports SG&A rather than a separate sales & marketing line; SG&A % of revenue shown.")
    if series["dna"][0] == "Depreciation":
        notes.append("Only a depreciation tag is reported (no combined D&A); EBITDA is operating income plus depreciation.")

    annual = {k: _annual_series(rows) for k, (_, rows) in series.items()}
    fy_ends = sorted(annual["revenue"])[-(years + 1) :]

    def val(key: str, end: str) -> float | None:
        r = annual[key].get(end)
        return float(r["val"]) if r else None

    def build(label: str, end: str, get) -> PeriodMetrics:
        rev = get("revenue")
        gp = get("gross_profit")
        if gp is None and get("cogs") is not None and rev is not None:
            gp = rev - get("cogs")
        cfo, capex = get("cfo"), get("capex")
        fcf = (cfo - abs(capex)) if cfo is not None and capex is not None else cfo
        opi, dna = get("operating_income"), get("dna")
        ebitda = (opi + dna) if opi is not None and dna is not None else None
        m = PeriodMetrics(
            label=label,
            period_end=end,
            revenue=rev,
            gross_margin_pct=_pct(gp, rev),
            operating_margin_pct=_pct(opi, rev),
            net_margin_pct=_pct(get("net_income"), rev),
            eps_diluted=round(get("eps_diluted"), 2) if get("eps_diluted") is not None else None,
            rnd_pct_revenue=_pct(get("rnd"), rev),
            snm_pct_revenue=_pct(get("snm"), rev),
            fcf=fcf,
            fcf_margin_pct=_pct(fcf, rev),
            ebitda=ebitda,
        )
        return m

    periods: list[PeriodMetrics] = []
    for i, end in enumerate(fy_ends):
        fy_label = f"FY{_d(end).year}" if _d(end).month >= 6 else f"FY{_d(end).year}"
        m = build(fy_label, end, lambda k, e=end: val(k, e))
        if i > 0:
            m.revenue_growth_pct = _growth(m.revenue, val("revenue", fy_ends[i - 1]))
        periods.append(m)
    periods = periods[-years:]

    ttm_vals: dict[str, float | None] = {}
    ttm_end, method = "", ""
    for key, (_, rows) in series.items():
        if key == "eps_diluted":
            continue
        v, end, meth = _ttm(rows, annual[key]) if rows else (None, "", "unavailable")
        ttm_vals[key] = v
        if key == "revenue":
            ttm_end, method = end, meth
    eps_ttm, _, _ = _ttm(series["eps_diluted"][1], annual["eps_diluted"]) if series["eps_diluted"][1] else (None, "", "")
    ttm_vals["eps_diluted"] = eps_ttm
    ttm: PeriodMetrics | None = None
    if ttm_vals.get("revenue") is not None:
        ttm = build("TTM", ttm_end, lambda k: ttm_vals.get(k))
        prior_rev = None
        if method == "fy_plus_ytd_delta":
            prior_rev = _ttm_prior_year(rev_rows, annual["revenue"], ttm_end)
        elif fy_ends and len(fy_ends) >= 2:
            prior_rev = val("revenue", fy_ends[-2])
        ttm.revenue_growth_pct = _growth(ttm.revenue, prior_rev)
        if ttm.revenue_growth_pct is not None and ttm.fcf_margin_pct is not None:
            ttm.rule_of_40 = round(ttm.revenue_growth_pct + ttm.fcf_margin_pct, 1)
        if method == "fiscal_year":
            notes.append("No 10-Q filed after the latest 10-K; TTM equals the latest fiscal year.")
    for m in periods:
        if m.revenue_growth_pct is not None and m.fcf_margin_pct is not None:
            m.rule_of_40 = round(m.revenue_growth_pct + m.fcf_margin_pct, 1)

    cash, cash_as_of = _latest_instant(_first_tag_with_data(company_facts, INSTANT_TAGS["cash"])[1])
    sti, _ = _latest_instant(_first_tag_with_data(company_facts, INSTANT_TAGS["short_term_investments"])[1])
    debt_nc, _ = _latest_instant(_first_tag_with_data(company_facts, INSTANT_TAGS["debt_noncurrent"])[1])
    debt_c, _ = _latest_instant(_first_tag_with_data(company_facts, INSTANT_TAGS["debt_current"])[1])
    total_debt = (debt_nc or 0.0) + (debt_c or 0.0) if (debt_nc is not None or debt_c is not None) else None
    cash_total = (cash or 0.0) + (sti or 0.0) if cash is not None else None
    shares, shares_as_of = _latest_instant_any(_first_tag_with_data(company_facts, SHARES_TAGS, taxonomy="dei")[1])
    if total_debt is None:
        notes.append("No debt tags found; treated as zero debt in enterprise value.")
    balance = BalanceSnapshot(
        cash_and_investments=cash_total,
        total_debt=total_debt,
        net_cash=(cash_total - (total_debt or 0.0)) if cash_total is not None else None,
        as_of=cash_as_of,
        shares_outstanding=shares,
        shares_as_of=shares_as_of,
    )
    return Financials(revenue_tag=rev_tag, snm_tag=series["snm"][0], periods=periods, ttm=ttm, balance=balance, notes=notes)


def _latest_instant_any(rows: list[dict[str, Any]]) -> tuple[float | None, str]:
    inst = [r for r in rows if "start" not in r]
    if not inst:
        return None, ""
    r = max(inst, key=lambda r: (r["end"], r.get("filed", "")))
    return float(r["val"]), r["end"]


def _ttm_prior_year(rev_rows: list[dict[str, Any]], annual: dict[str, dict[str, Any]], ttm_end: str) -> float | None:
    """TTM as of one year before ttm_end, computed with the same FY + YTD delta method."""
    target = _d(ttm_end).replace(year=_d(ttm_end).year - 1)
    fy_ends = [e for e in sorted(annual) if _d(e) <= target]
    if not fy_ends:
        return None
    fy_end = fy_ends[-1]
    fy_val = float(annual[fy_end]["val"])
    ytd = [r for r in _ytd_rows(rev_rows) if fy_end < r["end"] and abs((_d(r["end"]) - target).days) <= 10]
    if not ytd:
        return fy_val if abs((_d(fy_end) - target).days) <= 10 else None
    cur = max(ytd, key=lambda r: (_d(r["end"]) - _d(r["start"])).days)
    cur_days = (_d(cur["end"]) - _d(cur["start"])).days
    prior = [
        r
        for r in _ytd_rows(rev_rows)
        if abs((_d(cur["end"]) - _d(r["end"])).days - 365) <= 10 and abs((_d(r["end"]) - _d(r["start"])).days - cur_days) <= 10
    ]
    if not prior:
        return None
    return fy_val + float(cur["val"]) - float(prior[-1]["val"])


@dataclass
class Valuation:
    price: float
    market_cap: float | None
    enterprise_value: float | None
    ev_to_revenue: float | None
    ev_to_ebitda: float | None
    price_to_fcf: float | None
    price_to_earnings: float | None
    ev_to_revenue_3y_avg: float | None
    ev_to_revenue_3y_low: float | None
    ev_to_revenue_3y_high: float | None
    notes: list[str] = field(default_factory=list)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return round(num / den, 1)


def compute_valuation(fin: Financials, price: float, monthly_closes: list[tuple[str, float]]) -> Valuation:
    notes: list[str] = []
    shares = fin.balance.shares_outstanding
    ttm = fin.ttm
    mcap = price * shares if shares else None
    debt = fin.balance.total_debt or 0.0
    cash = fin.balance.cash_and_investments or 0.0
    ev = (mcap + debt - cash) if mcap is not None else None
    rev = ttm.revenue if ttm else None
    ebitda = ttm.ebitda if ttm else None
    fcf = ttm.fcf if ttm else None
    eps = ttm.eps_diluted if ttm else None
    if ebitda is None:
        notes.append("EBITDA unavailable (no depreciation & amortization tag); EV/EBITDA omitted.")
    hist: list[float] = []
    if shares and monthly_closes and fin.periods:
        annual_rev = sorted((p.period_end, p.revenue) for p in fin.periods if p.revenue)
        for d, close in monthly_closes:
            trailing = [r for end, r in annual_rev if end <= d]
            if not trailing:
                continue
            ev_t = close * shares + debt - cash
            hist.append(ev_t / trailing[-1])
        if hist:
            notes.append(
                "3-year EV/Revenue history uses current shares outstanding and net debt with month-end prices "
                "against the trailing fiscal-year revenue — an approximation of the historical multiple."
            )
    return Valuation(
        price=price,
        market_cap=mcap,
        enterprise_value=ev,
        ev_to_revenue=_ratio(ev, rev),
        ev_to_ebitda=_ratio(ev, ebitda),
        price_to_fcf=_ratio(mcap, fcf),
        price_to_earnings=_ratio(price, eps) if eps and eps > 0 else None,
        ev_to_revenue_3y_avg=round(sum(hist) / len(hist), 1) if hist else None,
        ev_to_revenue_3y_low=round(min(hist), 1) if hist else None,
        ev_to_revenue_3y_high=round(max(hist), 1) if hist else None,
        notes=notes,
    )
