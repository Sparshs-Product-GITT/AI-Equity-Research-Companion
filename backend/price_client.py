from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


class PriceUnavailable(Exception):
    pass


@dataclass
class Quote:
    symbol: str
    price: float
    currency: str
    exchange: str
    long_name: str
    fifty_two_week_low: float | None
    fifty_two_week_high: float | None
    as_of: str
    source_url: str
    monthly_closes: list[tuple[str, float]] = field(default_factory=list)


async def get_quote(client: httpx.AsyncClient, symbol: str, years: int = 3) -> Quote:
    url = CHART_URL.format(symbol=symbol.upper())
    resp = await client.get(url, params={"range": f"{years}y", "interval": "1mo"}, headers=_HEADERS, timeout=30.0)
    if resp.status_code != 200:
        raise PriceUnavailable(f"Yahoo chart returned {resp.status_code} for {symbol}")
    payload = resp.json().get("chart", {})
    results = payload.get("result")
    if not results:
        raise PriceUnavailable(f"No chart data for {symbol}: {payload.get('error')}")
    r = results[0]
    meta = r["meta"]
    stamps = r.get("timestamp") or []
    closes = (r.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
    monthly = [
        (datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"), float(c))
        for t, c in zip(stamps, closes)
        if c is not None
    ]
    as_of_ts = meta.get("regularMarketTime")
    as_of = datetime.fromtimestamp(as_of_ts, tz=timezone.utc).strftime("%Y-%m-%d") if as_of_ts else ""
    return Quote(
        symbol=symbol.upper(),
        price=float(meta["regularMarketPrice"]),
        currency=meta.get("currency", "USD"),
        exchange=meta.get("fullExchangeName", ""),
        long_name=meta.get("longName") or meta.get("shortName") or symbol.upper(),
        fifty_two_week_low=meta.get("fiftyTwoWeekLow"),
        fifty_two_week_high=meta.get("fiftyTwoWeekHigh"),
        as_of=as_of,
        source_url=f"https://finance.yahoo.com/quote/{symbol.upper()}",
        monthly_closes=monthly,
    )
