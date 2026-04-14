"""
InsightEngine — translates KPI data into AI-generated narrative insights.
Author: Tuan Tran
"""

from __future__ import annotations

import logging
from typing import Generator, Dict, Any

from src.ai.qwen_client import QwenAIClient

logger = logging.getLogger(__name__)


class InsightEngine:
    """
    Produces short, punchy AI-written insights for each KPI card
    and chart section displayed in the dashboard.
    """

    def __init__(self, ai_client: QwenAIClient | None = None) -> None:
        self._ai = ai_client or QwenAIClient()

    def kpi_insights(self, kpi_data: Dict[str, Any]) -> Generator[str, None, None]:
        """Generate dashboard KPI commentary."""
        yield from self._ai.get_kpi_insights(kpi_data)

    def trend_commentary(
        self, metric_name: str, values: list, labels: list
    ) -> Generator[str, None, None]:
        """Narrate a time-series trend for a single metric."""
        context = {
            "metric": metric_name,
            "period_labels": labels,
            "values": values,
            "latest": values[-1] if values else None,
            "change_vs_prior": (
                round((values[-1] - values[-2]) / values[-2] * 100, 1)
                if len(values) >= 2 and values[-2] != 0
                else None
            ),
        }
        yield from self._ai.get_kpi_insights(context)

    def regional_insights(
        self, region_data: Dict[str, float]
    ) -> Generator[str, None, None]:
        """Highlight top and bottom performing regions."""
        sorted_regions = sorted(region_data.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_regions[:3]
        bottom3 = sorted_regions[-3:]
        context = {
            "analysis_type": "Regional Performance",
            "top_performers": dict(top3),
            "bottom_performers": dict(bottom3),
            "total_regions": len(region_data),
        }
        yield from self._ai.get_kpi_insights(context)
