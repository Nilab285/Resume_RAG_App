
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "resumes.db"

# ==========================================================
# LLM Configuration
# ==========================================================

GROQ_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Please configure your .env file."
    )

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama-3.1-8b-instant"
)

MAX_CONCURRENT_LLM = 3
LLM_MAX_RETRIES = 5
LLM_INITIAL_BACKOFF = 1

# ==========================================================
# SQLite
# ==========================================================

SQLITE_BUSY_TIMEOUT = 10000

# ==========================================================
# Parallel Processing
# ==========================================================

MAX_INGESTION_WORKERS = 4
MAX_EXTRACTION_WORKERS = 3