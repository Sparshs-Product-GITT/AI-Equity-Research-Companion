from dataclasses import dataclass, field
from datetime import date

from .edgar_client import Company, TenKSections
from .financial_math import Financials, Valuation
from .price_client import Quote


def fmt_usd(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:.{decimals}f}T"
    if a >= 1e9:
        return f"${v / 1e9:.{decimals}f}B"
    if a >= 1e6:
        return f"${v / 1e6:.{decimals}f}M"
    return f"${v:,.0f}"


def fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def fmt_x(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}x"


@dataclass
class RunContext:
    ticker: str
    focus_note: str
    company: Company
    quote: Quote
    financials: Financials
    valuation: Valuation
    tenk: TenKSections
    last_8k_date: str | None
    today: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def header(self) -> str:
        c, q = self.company, self.quote
        return (
            f"Company: {c.name} ({self.ticker}, {q.exchange or ', '.join(c.exchanges or [])})\n"
            f"SEC CIK: {c.cik} | SIC: {c.sic_description} | Fiscal year end: month/day {c.fiscal_year_end}\n"
            f"Price: ${q.price:.2f} as of {q.as_of} | 52-week range ${q.fifty_two_week_low}-${q.fifty_two_week_high}\n"
            f"Today's date: {self.today}"
            + (f"\nAnalyst focus note: {self.focus_note}" if self.focus_note else "")
        )

    def financials_brief(self) -> str:
        fin, val = self.financials, self.valuation
        rows = fin.periods + ([fin.ttm] if fin.ttm else [])
        lines = ["Period | Revenue | YoY growth | Gross margin | Op margin | Net margin | FCF margin | Rule of 40 | R&D % rev | S&M % rev | Diluted EPS"]
        for m in rows:
            lines.append(
                f"{m.label} (to {m.period_end}) | {fmt_usd(m.revenue)} | {fmt_pct(m.revenue_growth_pct)} | {fmt_pct(m.gross_margin_pct)} | "
                f"{fmt_pct(m.operating_margin_pct)} | {fmt_pct(m.net_margin_pct)} | {fmt_pct(m.fcf_margin_pct)} | "
                f"{'n/a' if m.rule_of_40 is None else m.rule_of_40} | {fmt_pct(m.rnd_pct_revenue)} | {fmt_pct(m.snm_pct_revenue)} | "
                f"{'n/a' if m.eps_diluted is None else m.eps_diluted}"
            )
        b = fin.balance
        lines.append(
            f"Balance sheet (as of {b.as_of}): cash & short-term investments {fmt_usd(b.cash_and_investments)}, "
            f"total debt {fmt_usd(b.total_debt)}, net cash {fmt_usd(b.net_cash)}; shares outstanding "
            f"{'n/a' if b.shares_outstanding is None else f'{b.shares_outstanding / 1e6:,.0f}M'} (as of {b.shares_as_of})"
        )
        lines.append(
            f"Valuation (trailing, price ${val.price:.2f}): market cap {fmt_usd(val.market_cap)}, EV {fmt_usd(val.enterprise_value)}, "
            f"EV/Revenue {fmt_x(val.ev_to_revenue)}, EV/EBITDA {fmt_x(val.ev_to_ebitda)}, P/FCF {fmt_x(val.price_to_fcf)}, "
            f"P/E {fmt_x(val.price_to_earnings)}; 3-year EV/Revenue range {fmt_x(val.ev_to_revenue_3y_low)}-{fmt_x(val.ev_to_revenue_3y_high)}, "
            f"average {fmt_x(val.ev_to_revenue_3y_avg)}"
        )
        for n in fin.notes + val.notes:
            lines.append(f"Note: {n}")
        return "\n".join(lines)
