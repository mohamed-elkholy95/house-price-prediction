"""
Configuration Module
====================

Centralizes all configurable parameters for the house price prediction pipeline.

Environment variables override defaults, following the 12-factor app pattern:
- RANDOM_SEED: Reproducibility seed for all random operations
- API_HOST: FastAPI bind address
- API_PORT: FastAPI listen port
- LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

This makes the project deployable across environments (dev, staging, prod)
without code changes.
"""

import logging
import os
from pathlib import Path

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- ML Settings ---
# Random seed ensures reproducible results across runs. Set to any integer;
# the specific value (42) is a convention from "The Hitchhiker's Guide to
# the Galaxy" and has no mathematical significance.
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# --- API Settings ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8012"))

# --- Data Generation Defaults ---
DEFAULT_N_SAMPLES = int(os.getenv("DEFAULT_N_SAMPLES", "1000"))
DEFAULT_TEST_SIZE = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
