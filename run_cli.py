"""Run the full pipeline from the command line: python run_cli.py CRM --focus "margins" """
import argparse
import asyncio
import sys

from backend import orchestrator


async def main(ticker: str, focus: str) -> int:
    p = orchestrator.Pipeline(ticker, focus)
    print(f"run {p.state.run_id} -> {p.run_dir}")

    async def watch():
        last: dict[str, str] = {}
        while p.state.status == "running":
            for name, s in p.state.steps.items():
                key = f"{s.status}|{s.detail}"
                if last.get(name) != key and s.status != "pending":
                    last[name] = key
                    print(f"  [{s.status:8}] {s.title}{' — ' + s.detail if s.detail else ''}")
            await asyncio.sleep(1.5)

    watcher = asyncio.create_task(watch())
    state = await p.run()
    await asyncio.sleep(1.6)
    watcher.cancel()
    print(f"\nstatus: {state.status}{' (partial)' if state.partial else ''} · cost ${state.total_cost_usd:.2f}")
    if state.error:
        print("error:", state.error)
    if state.qa:
        print("QA:", "passed" if state.qa["passed"] else "flagged", "—", state.qa["summary"])
    if state.final_html:
        print("one-pager:", p.run_dir / state.final_html)
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--focus", default="")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.ticker, a.focus)))
