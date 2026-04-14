"""
AnomalyDetector — statistical detection + AI narrative explanation.
Author: Tuan Tran
"""

from __future__ import annotations

import logging
from typing import Generator, List, Dict, Any

import numpy as np
import pandas as pd

from src.ai.qwen_client import QwenAIClient

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Two-stage anomaly detection:
    1. Statistical stage  — Z-score and IQR-based flagging.
    2. AI narrative stage — Qwen 3 explains each anomaly in business terms.
    """

    Z_THRESHOLD = 2.5   # |z| > 2.5 → flag
    IQR_MULTIPLIER = 1.5

    def __init__(self, ai_client: QwenAIClient | None = None) -> None:
        self._ai = ai_client or QwenAIClient()

    # ── Statistical detection ─────────────────────────────────
    def detect(self, series: pd.Series, metric_name: str) -> List[Dict[str, Any]]:
        """
        Run Z-score and IQR detection on a numeric series.
        Returns a list of anomaly dicts (may be empty).
        """
        anomalies: List[Dict[str, Any]] = []
        if len(series) < 4:
            return anomalies

        clean = series.dropna()
        mean, std = clean.mean(), clean.std()
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1

        for idx, val in series.items():
            if pd.isna(val):
                continue
            z = (val - mean) / std if std > 0 else 0
            iqr_flag = val < (q1 - self.IQR_MULTIPLIER * iqr) or val > (q3 + self.IQR_MULTIPLIER * iqr)
            z_flag = abs(z) > self.Z_THRESHOLD

            if z_flag or iqr_flag:
                severity = self._severity(abs(z))
                anomalies.append(
                    {
                        "metric": metric_name,
                        "period": str(idx),
                        "value": round(float(val), 2),
                        "mean": round(float(mean), 2),
                        "z_score": round(float(z), 2),
                        "direction": "above" if val > mean else "below",
                        "severity": severity,
                        "detection_method": "Z-score" if z_flag else "IQR",
                    }
                )
        return anomalies

    def detect_dataframe(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Run detection across multiple columns of a DataFrame."""
        all_anomalies: List[Dict[str, Any]] = []
        for col in numeric_cols:
            if col in df.columns:
                all_anomalies.extend(self.detect(df[col], col))
        return all_anomalies

    # ── AI explanation ────────────────────────────────────────
    def explain(
        self, anomalies: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Stream AI-generated explanations for a list of anomalies."""
        if not anomalies:
            yield "✅ No statistical anomalies detected in the selected dataset."
            return
        yield from self._ai.explain_anomalies(anomalies)

    def detect_and_explain(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> Generator[str, None, None]:
        """Combined: detect anomalies then explain them in one streaming call."""
        anomalies = self.detect_dataframe(df, numeric_cols)
        yield from self.explain(anomalies)

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _severity(abs_z: float) -> str:
        if abs_z >= 4.0:
            return "CRITICAL"
        if abs_z >= 3.0:
            return "HIGH"
        if abs_z >= 2.5:
            return "MEDIUM"
        return "LOW"
