"""Re-render an existing run's one-pager from its saved section outputs (no LLM calls).
Refetches EDGAR/price data (seconds, free) so template changes can be iterated cheaply.

    python recompose.py <run_id>
"""
import asyncio
import json
import sys

import httpx

from backend import compositor, orchestrator


async def main(run_id: str) -> int:
    st = orchestrator.load_state(run_id)
    if st is None:
        print(f"no run {run_id}")
        return 1
    p = orchestrator.Pipeline(st["ticker"], st.get("focus_note", ""), run_id=run_id)
    p.state.qa = st.get("qa")
    async with httpx.AsyncClient(follow_redirects=True) as http:
        await p._data_fetch(http)
    sections_dir = p.run_dir / "sections"
    for f in sections_dir.glob("*.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        if payload["agent"] != "qa_review":
            p.sections[payload["agent"]] = payload["data"]
    pm = p.run_dir / "data" / "peer_metrics.json"
    if pm.exists():
        p.sections["peer_metrics"] = json.loads(pm.read_text(encoding="utf-8"))
    html, problems, over = compositor.compose(p.ctx, p.sections, {**st, "qa": p.state.qa})
    (p.run_dir / "onepager.html").write_text(html, encoding="utf-8")
    print("structural:", problems or "ok")
    print("over budget:", len(over))
    for o in over:
        print("  ", o)
    print("wrote", p.run_dir / "onepager.html")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1])))
