import asyncio
import html
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import SEC_USER_AGENT

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_ticker_cache: dict[str, dict[str, Any]] | None = None
_lock = asyncio.Lock()


class TickerNotFound(Exception):
    pass


@dataclass
class Filing:
    form: str
    accession: str
    primary_doc: str
    filing_date: str
    report_date: str
    cik: int

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_doc)

    @property
    def index_url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}/"


@dataclass
class Company:
    ticker: str
    cik: int
    name: str
    sic_description: str = ""
    fiscal_year_end: str = ""
    exchanges: list[str] | None = None
    filings: list[Filing] | None = None

    def latest(self, *forms: str) -> Filing | None:
        for f in self.filings or []:
            if f.form in forms:
                return f
        return None


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    for attempt in range(3):
        resp = await client.get(url, headers=_HEADERS, timeout=30.0)
        if resp.status_code == 429 or resp.status_code >= 500:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


async def _ticker_map(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    global _ticker_cache
    async with _lock:
        if _ticker_cache is None:
            data = (await _get(client, TICKER_MAP_URL)).json()
            _ticker_cache = {v["ticker"].upper(): v for v in data.values()}
    return _ticker_cache


async def resolve_company(client: httpx.AsyncClient, ticker: str) -> Company:
    ticker = ticker.strip().upper()
    entry = (await _ticker_map(client)).get(ticker)
    if not entry:
        raise TickerNotFound(f"Ticker {ticker!r} not found in SEC EDGAR company list")
    cik = int(entry["cik_str"])
    sub = (await _get(client, SUBMISSIONS_URL.format(cik=cik))).json()
    recent = sub.get("filings", {}).get("recent", {})
    filings = [
        Filing(
            form=recent["form"][i],
            accession=recent["accessionNumber"][i],
            primary_doc=recent["primaryDocument"][i],
            filing_date=recent["filingDate"][i],
            report_date=recent["reportDate"][i],
            cik=cik,
        )
        for i in range(len(recent.get("form", [])))
    ]
    return Company(
        ticker=ticker,
        cik=cik,
        name=sub.get("name") or entry["title"],
        sic_description=sub.get("sicDescription", ""),
        fiscal_year_end=sub.get("fiscalYearEnd", ""),
        exchanges=sub.get("exchanges") or [],
        filings=filings,
    )


async def get_company_facts(client: httpx.AsyncClient, cik: int) -> dict[str, Any]:
    return (await _get(client, FACTS_URL.format(cik=cik))).json()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_NL_RE = re.compile(r"\n{3,}")


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", raw)
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _NL_RE.sub("\n\n", text).strip()


_PAGE_NO_RE = re.compile(r"^\s*\d{1,3}\s*$")


def _is_toc_entry(text: str, heading_end: int) -> bool:
    """A heading followed by a bare page number is a table-of-contents entry, not the section."""
    following = text[heading_end : heading_end + 40].lstrip("\n ")
    first_line = following.split("\n", 1)[0]
    return bool(_PAGE_NO_RE.match(first_line))


def _extract_item(text: str, start_pat: str, end_pat: str) -> str:
    best = ""
    for m in re.finditer(r"(?im)^[ \t]*" + start_pat, text):
        if _is_toc_entry(text, m.end()):
            continue
        m_end = re.search(r"(?im)^[ \t]*" + end_pat, text[m.end() :])
        e = m.end() + m_end.start() if m_end else min(len(text), m.start() + 120_000)
        chunk = text[m.start() : e]
        if len(chunk) > len(best):
            best = chunk
    return best.strip()


@dataclass
class TenKSections:
    url: str
    business: str
    risk_factors: str
    mdna: str


def _fuzzy(*words: str) -> str:
    """Heading words with optional whitespace between letters — some filers (e.g. Microsoft)
    emit 'UNRESOLVE D STAFF COMMENTS' after HTML-to-text conversion."""
    return r"\s*".join(r"\s*".join(re.escape(ch) for ch in w) for w in words)


_ITEM = {
    "1": r"item\s*1\s*\.?\s*" + _fuzzy("business"),
    "1a": r"item\s*1a\s*\.?\s*" + _fuzzy("risk", "factors"),
    "1b_or_2": r"item\s*1b\s*\.?\s*" + _fuzzy("unresolved") + r"|item\s*2\s*\.?\s*" + _fuzzy("properties"),
    "7": r"item\s*7\s*\.?\s*" + _fuzzy("management") + r".{0,3}s\s*" + _fuzzy("discussion"),
    "7a_or_8": r"item\s*7a\s*\.?\s*" + _fuzzy("quantitative") + r"|item\s*8\s*\.?\s*" + _fuzzy("financial", "statements"),
}


async def get_10k_sections(client: httpx.AsyncClient, filing: Filing, max_chars: int = 40_000) -> TenKSections:
    raw = (await _get(client, filing.url)).text
    text = _html_to_text(raw)
    business = _extract_item(text, _ITEM["1"], _ITEM["1a"])
    risks = _extract_item(text, _ITEM["1a"], _ITEM["1b_or_2"])
    mdna = _extract_item(text, _ITEM["7"], _ITEM["7a_or_8"])
    return TenKSections(
        url=filing.url,
        business=business[:max_chars],
        risk_factors=risks[:max_chars],
        mdna=mdna[:max_chars],
    )


def latest_earnings_8k(company: Company) -> Filing | None:
    """Most recent 8-K, used as the best free proxy for the last earnings release date.

    Item 2.02 8-Ks aren't tagged in the submissions feed, so the caller should
    cross-check the date against the news agent's findings.
    """
    return company.latest("8-K")
