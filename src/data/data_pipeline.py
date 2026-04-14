"""
DataPipeline — loads, validates, and aggregates mock CSV datasets.

Returns domain DataFrames plus pre-computed KPI summaries.
Author: Tuan Tran
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from src.config.settings import Config
from src.data.mock_data_generator import MockDataGenerator

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Singleton-friendly data layer.

    Usage
    -----
    pipeline = DataPipeline()
    df_sales = pipeline.sales
    kpis     = pipeline.kpis
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir or Config.DATA_DIR
        self._ensure_data()

    # ── Raw DataFrames ────────────────────────────────────────
    @cached_property
    def sales(self) -> pd.DataFrame:
        df = pd.read_csv(self._dir / "sales_data.csv", parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def finance(self) -> pd.DataFrame:
        df = pd.read_csv(self._dir / "financial_data.csv", parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def hr(self) -> pd.DataFrame:
        df = pd.read_csv(self._dir / "hr_data.csv", parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def operations(self) -> pd.DataFrame:
        df = pd.read_csv(self._dir / "operations_data.csv", parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── Monthly aggregations ──────────────────────────────────
    @cached_property
    def sales_monthly(self) -> pd.DataFrame:
        return (
            self.sales
            .groupby(pd.Grouper(key="date", freq="MS"))
            .agg(
                revenue=("revenue", "sum"),
                units_sold=("units_sold", "sum"),
                gross_profit=("gross_profit", "sum"),
                avg_discount=("discount_pct", "mean"),
            )
            .reset_index()
        )

    @cached_property
    def sales_by_region(self) -> pd.DataFrame:
        return (
            self.sales
            .groupby("region")
            .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

    @cached_property
    def sales_by_product(self) -> pd.DataFrame:
        return (
            self.sales
            .groupby("product")
            .agg(
                revenue=("revenue", "sum"),
                units_sold=("units_sold", "sum"),
                gross_profit=("gross_profit", "sum"),
            )
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

    @cached_property
    def sales_by_channel(self) -> pd.DataFrame:
        return (
            self.sales
            .groupby("channel")["revenue"]
            .sum()
            .reset_index()
        )

    @cached_property
    def hr_monthly(self) -> pd.DataFrame:
        return (
            self.hr
            .groupby("date")
            .agg(
                headcount=("headcount", "sum"),
                new_hires=("new_hires", "sum"),
                attritions=("attritions", "sum"),
                attrition_rate=("attrition_rate_pct", "mean"),
                training_hours=("training_hours", "mean"),
            )
            .reset_index()
        )

    # ── Top-level KPIs ────────────────────────────────────────
    @cached_property
    def kpis(self) -> Dict[str, Any]:
        fin = self.finance
        sal = self.sales_monthly
        hr_m = self.hr_monthly
        ops = self.operations

        # Latest month
        latest_fin = fin.iloc[-1]
        latest_sal = sal.iloc[-1]
        latest_hr = hr_m.iloc[-1]
        latest_ops = ops.iloc[-1]

        # Prior month for delta
        prior_fin = fin.iloc[-2] if len(fin) > 1 else latest_fin
        prior_sal = sal.iloc[-2] if len(sal) > 1 else latest_sal

        def pct_chg(curr, prev):
            return round((curr - prev) / max(abs(prev), 1) * 100, 1)

        total_revenue = round(fin["revenue"].sum(), 0)
        total_profit  = round(fin["net_profit"].sum(), 0)
        gross_margin  = round(fin["gross_profit"].mean() / fin["revenue"].mean() * 100, 1)

        return {
            # Finance
            "total_revenue":       total_revenue,
            "revenue_delta_pct":   pct_chg(latest_fin["revenue"], prior_fin["revenue"]),
            "total_net_profit":    total_profit,
            "profit_delta_pct":    pct_chg(latest_fin["net_profit"], prior_fin["net_profit"]),
            "gross_margin_pct":    gross_margin,
            "ebitda_latest":       round(latest_fin["ebitda"], 0),
            # Sales
            "total_units_sold":    int(self.sales["units_sold"].sum()),
            "top_product":         self.sales_by_product.iloc[0]["product"],
            "top_region":          self.sales_by_region.iloc[0]["region"],
            "revenue_vs_budget_pct": pct_chg(
                latest_fin["revenue"], latest_fin["budget_revenue"]
            ),
            # HR
            "total_headcount":     int(latest_hr["headcount"]),
            "headcount_delta":     int(latest_hr["headcount"] - hr_m.iloc[-2]["headcount"])
                                   if len(hr_m) > 1 else 0,
            "attrition_rate_pct":  round(float(latest_hr["attrition_rate"]), 2),
            "new_hires_ytd":       int(self.hr["new_hires"].tail(12).sum()),
            # Operations
            "efficiency_pct":      round(float(latest_ops["process_efficiency_pct"]), 2),
            "sla_compliance_pct":  round(float(latest_ops["sla_compliance_pct"]), 2),
            "defect_rate_pct":     round(float(latest_ops["defect_rate_pct"]), 2),
            "avg_resolution_hrs":  round(float(latest_ops["avg_resolution_hours"]), 1),
        }

    # ── Serialisable summary for AI ───────────────────────────
    def ai_summary(self) -> Dict[str, Any]:
        """Compact dict suitable for injection into LLM prompts (token-efficient)."""
        # Last 3 months only to reduce token count
        fin_tail = self.finance.tail(3)[
            ["date", "revenue", "net_profit", "ebitda"]
        ].to_dict("records")

        ops_tail = self.operations.tail(3)[
            ["date", "process_efficiency_pct", "sla_compliance_pct", "defect_rate_pct"]
        ].to_dict("records")

        hr_tail = self.hr_monthly.tail(3)[
            ["date", "headcount", "attrition_rate"]
        ].to_dict("records")

        return {
            "kpis": self.kpis,
            "finance_last_3": fin_tail,
            "operations_last_3": ops_tail,
            "hr_last_3": hr_tail,
            "top_3_products": self.sales_by_product.head(3)[["product", "revenue"]].to_dict("records"),
            "top_3_regions": self.sales_by_region.head(3).to_dict("records"),
        }

    # ── Private ───────────────────────────────────────────────
    def _ensure_data(self) -> None:
        """Auto-generate data if CSV files are missing."""
        required = [
            "sales_data.csv",
            "financial_data.csv",
            "hr_data.csv",
            "operations_data.csv",
        ]
        if not all((self._dir / f).exists() for f in required):
            logger.info("Mock data not found — generating now…")
            MockDataGenerator(self._dir).generate_all()
