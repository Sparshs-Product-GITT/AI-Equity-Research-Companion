import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
TEMPLATE_DIR = ROOT / "backend" / "templates"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # The project .env is explicit configuration, so it wins over whatever the shell
        # inherited (e.g. a desktop-app session that pre-sets ANTHROPIC_BASE_URL).
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv(ROOT / ".env")

# Everything the Claude Code subprocess needs to reach the firm gateway.
ANTHROPIC_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
ANTHROPIC_ENV = {k: os.environ[k] for k in ANTHROPIC_ENV_KEYS if os.environ.get(k)}
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "EquityResearchCompanion/0.1 (set SEC_USER_AGENT in .env)"
)
MODEL = os.environ.get("ER_MODEL", "claude-opus-5")
MAX_BUDGET_USD = float(os.environ.get("ER_MAX_BUDGET_USD", "3.0"))
HAS_API_KEY = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"))


def _version_key(p: Path) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in p.parent.name.split("."))
    except ValueError:
        return (0,)


def find_claude_cli() -> str | None:
    """The Agent SDK wheel doesn't bundle claude.exe on every platform; fall back to the
    Claude desktop app's copy or the native installer's."""
    explicit = os.environ.get("CLAUDE_CLI_PATH") or os.environ.get("CLAUDE_CODE_EXECPATH")
    if explicit and Path(explicit).is_file():
        return explicit
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates += Path(local).glob("Packages/Claude_*/LocalCache/Roaming/Claude/claude-code/*/claude.exe")
    candidates += [Path.home() / ".local" / "bin" / "claude.exe", Path.home() / ".local" / "bin" / "claude"]
    existing = [c for c in candidates if c.is_file()]
    if not existing:
        return None
    return str(max(existing, key=_version_key))


CLI_PATH = find_claude_cli()
