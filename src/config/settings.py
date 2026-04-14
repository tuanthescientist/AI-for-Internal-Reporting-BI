"""
Centralised configuration — reads from .env / environment variables.
Author: Tuan Tran
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Config:
    # ── Project ──────────────────────────────────────────────
    ROOT_DIR: Path = ROOT_DIR
    APP_NAME: str = os.getenv("APP_NAME", "AI Internal Reporting & BI Platform")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    AUTHOR: str = "Tuan Tran"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # ── AI / Groq ─────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    QWEN_MODEL: str = "qwen/qwen3-32b"
    AI_TEMPERATURE: float = 0.6
    AI_MAX_TOKENS: int = 4096
    AI_TOP_P: float = 0.95
    AI_REASONING_EFFORT: str = "default"

    # ── Data Paths ────────────────────────────────────────────
    DATA_DIR: Path = ROOT_DIR / os.getenv("DATA_DIR", "data/mock")
    EXPORT_DIR: Path = ROOT_DIR / os.getenv("EXPORT_DIR", "data/exports")

    # ── UI ────────────────────────────────────────────────────
    WINDOW_TITLE: str = f"{APP_NAME} v{APP_VERSION}"
    WINDOW_MIN_WIDTH: int = 1_400
    WINDOW_MIN_HEIGHT: int = 860
    ACCENT_COLOR: str = "#00D4FF"
    BG_DARK: str = "#0D1117"
    BG_CARD: str = "#161B22"
    BG_PANEL: str = "#1C2128"
    TEXT_PRIMARY: str = "#E6EDF3"
    TEXT_SECONDARY: str = "#8B949E"
    SUCCESS_COLOR: str = "#3FB950"
    WARNING_COLOR: str = "#D29922"
    DANGER_COLOR: str = "#F85149"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
