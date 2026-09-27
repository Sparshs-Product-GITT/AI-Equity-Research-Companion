import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import orchestrator
from .config import BASE_URL, CLI_PATH, HAS_API_KEY, MODEL, OUTPUT_DIR

app = FastAPI(title="Equity Research Companion", version="0.1.0")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

_active: dict[str, asyncio.Task] = {}
_pipelines: dict[str, orchestrator.Pipeline] = {}
_TICKER_RE = re.compile(r"^[A-Za-z.\-]{1,10}$")


class RunRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    focus_note: str = Field(default="", max_length=500)


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index() -> str:
    return FRONTEND.read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "has_credentials": HAS_API_KEY,
        "cli_found": CLI_PATH is not None,
        "base_url": BASE_URL,
        "active_runs": [rid for rid, t in _active.items() if not t.done()],
    }


@app.get("/api/runs")
async def runs() -> list[dict]:
    return orchestrator.list_runs()


@app.post("/api/runs", status_code=202)
async def start_run(req: RunRequest) -> dict:
    if not HAS_API_KEY:
        raise HTTPException(503, "No ANTHROPIC_API_KEY configured — copy env.example to .env and set it")
    if not _TICKER_RE.match(req.ticker.strip()):
        raise HTTPException(422, "Ticker must be 1-10 letters")
    if any(not t.done() for t in _active.values()):
        raise HTTPException(409, "A run is already in progress; this MVP processes one ticker at a time")
    pipeline = orchestrator.Pipeline(req.ticker, req.focus_note)
    _pipelines[pipeline.state.run_id] = pipeline
    _active[pipeline.state.run_id] = asyncio.create_task(pipeline.run())
    return {"run_id": pipeline.state.run_id}


@app.get("/api/runs/{run_id}")
async def run_status(run_id: str) -> dict:
    p = _pipelines.get(run_id)
    if p is not None:
        return p.state.to_dict()
    st = orchestrator.load_state(run_id)
    if st is None:
        raise HTTPException(404, "Unknown run")
    return st


def _safe_file(run_id: str, rel: str) -> Path:
    if not re.match(r"^[a-f0-9]{12}$", run_id):
        raise HTTPException(404, "Unknown run")
    base = (OUTPUT_DIR / run_id).resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found")
    return target


@app.get("/api/runs/{run_id}/onepager", response_class=HTMLResponse)
async def onepager(run_id: str) -> str:
    return _safe_file(run_id, "onepager.html").read_text(encoding="utf-8")


@app.get("/api/runs/{run_id}/download/onepager")
async def download_onepager(run_id: str) -> FileResponse:
    st = orchestrator.load_state(run_id) or {}
    fname = f"{st.get('ticker', 'onepager')}_onepager_{run_id}.html"
    return FileResponse(_safe_file(run_id, "onepager.html"), media_type="text/html", filename=fname)


@app.get("/api/runs/{run_id}/files/{path:path}")
async def run_file(run_id: str, path: str):
    target = _safe_file(run_id, path)
    media = {"json": "application/json", "md": "text/markdown", "html": "text/html"}.get(target.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(target, media_type=media, filename=f"{run_id}_{target.name}")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
