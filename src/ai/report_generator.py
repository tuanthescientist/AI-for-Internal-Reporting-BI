"""
ReportGenerator — orchestrates multi-domain AI report generation.
Author: Tuan Tran
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Generator, Dict, Any

from src.ai.qwen_client import QwenAIClient
from src.config.settings import Config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Coordinates data summarisation and AI narrative generation
    to produce structured business reports.
    """

    def __init__(self, ai_client: QwenAIClient | None = None) -> None:
        self._ai = ai_client or QwenAIClient()

    # ── Public ────────────────────────────────────────────────
    def executive_report(
        self, all_data_summary: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Stream a full multi-domain executive report."""
        summary = self._enrich_summary(all_data_summary)
        yield from self._ai.generate_executive_report(summary)

    def department_report(
        self, department: str, department_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Stream a department-scoped report."""
        yield from self._ai.generate_department_report(department, department_data)

    def monthly_summary(
        self, month_label: str, metrics: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Stream a concise monthly performance snapshot."""
        enriched = {"period": month_label, "metrics": metrics, "generated_at": str(datetime.now())}
        yield from self._ai.generate_executive_report(enriched)

    # ── Private helpers ───────────────────────────────────────
    @staticmethod
    def _enrich_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata before sending to AI."""
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "author": Config.AUTHOR,
            "platform": Config.APP_NAME,
            **summary,
        }
