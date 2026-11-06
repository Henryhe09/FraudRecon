"""Fraud Procedure Extraction settings."""
import os
from pathlib import Path

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
INPUT_PATH = None
MODEL_NAME = os.getenv("OPENAI_MODEL", "")
OUTPUT_DIR = Path("fraud_results_chunks")
OUTPUT_FILE_PREFIX = "fraud_results"
CHUNK_SIZE = 5000
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2
MAX_WORKERS = 5
