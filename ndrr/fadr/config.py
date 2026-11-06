import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# =========================
# API / Model
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "")
OPENAI_TIMEOUT_SECONDS = _env_float("OPENAI_TIMEOUT_SECONDS", 120.0)


# =========================
# IO defaults
# =========================
INPUT_PATTERN = os.getenv("RECON_INPUT_PATTERN", "")
OUTPUT_JSONL = os.getenv("RECON_OUTPUT_JSONL", "")
FULL_OUTPUT_DIR = os.getenv("RECON_FULL_OUTPUT_DIR", "")


# =========================
# Runtime defaults
# =========================
WORKERS = _env_int("RECON_WORKERS", 2)
MAX_CASES = _env_int("RECON_MAX_CASES", 0)


# =========================
# Reconstruction defaults
# =========================
TEMPERATURE = _env_float("RECON_TEMPERATURE", 0.7)
MANAGER_TEMPERATURE = _env_float("RECON_MANAGER_TEMPERATURE", 0.2)
MAX_TOKENS = _env_int("RECON_MAX_TOKENS", 900)
MAX_RETRIES = _env_int("RECON_MAX_RETRIES", 3)
RETRY_DELAY = _env_float("RECON_RETRY_DELAY", 2.0)
TAIL_SIZE = _env_int("RECON_TAIL_SIZE", 4)
MAX_TURNS_PER_PHASE = _env_int("RECON_MAX_TURNS_PER_PHASE", 5)
