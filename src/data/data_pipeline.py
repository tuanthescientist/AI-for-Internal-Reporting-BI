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
        # New schema: net_revenue, gross_revenue, gross_margin_pct
        rev_col = "net_revenue" if "net_revenue" in self.sales.columns else "gross_revenue"
        agg: Dict[str, Any] = {
            "revenue": (rev_col, "sum"),
            "units_sold": ("units_sold", "sum"),
        }
        if "gross_revenue" in self.sales.columns:
            agg["gross_revenue"] = ("gross_revenue", "sum")
        if "gross_margin_pct" in self.sales.columns:
            agg["gross_margin_pct"] = ("gross_margin_pct", "mean")
        return (
            self.sales
            .groupby(pd.Grouper(key="date", freq="MS"))
            .agg(**agg)
            .reset_index()
        )

    @cached_property
    def sales_by_region(self) -> pd.DataFrame:
        rev_col = "net_revenue" if "net_revenue" in self.sales.columns else "gross_revenue"
        return (
            self.sales
            .groupby("region")
            .agg(revenue=(rev_col, "sum"), units_sold=("units_sold", "sum"))
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

    @cached_property
    def sales_by_product(self) -> pd.DataFrame:
        rev_col = "net_revenue" if "net_revenue" in self.sales.columns else "gross_revenue"
        return (
            self.sales
            .groupby("product")
            .agg(revenue=(rev_col, "sum"), units_sold=("units_sold", "sum"))
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

    @cached_property
    def sales_by_channel(self) -> pd.DataFrame:
        rev_col = "net_revenue" if "net_revenue" in self.sales.columns else "gross_revenue"
        return (
            self.sales
            .groupby("channel")[rev_col]
            .sum()
            .reset_index()
            .rename(columns={rev_col: "revenue"})
        )

    @cached_property
    def hr_monthly(self) -> pd.DataFrame:
        hr = self.hr
        # Support both old and new schema column names
        hc_col  = "headcount_eop" if "headcount_eop" in hr.columns else "headcount"
        att_col = "attritions"    if "attritions"    in hr.columns else "attritions"
        att_rate_col = "attrition_rate_pct" if "attrition_rate_pct" in hr.columns else "attrition_rate"

        agg: Dict[str, Any] = {
            "headcount":      (hc_col, "sum"),
            "new_hires":      ("new_hires", "sum"),
            "attrition_rate": (att_rate_col, "mean"),
        }
        if att_col in hr.columns:
            agg["attritions"] = (att_col, "sum")
        if "training_hours" in hr.columns:
            agg["training_hours"] = ("training_hours", "mean")
        if "eNPS" in hr.columns:
            agg["eNPS"] = ("eNPS", "mean")
        if "engagement_score" in hr.columns:
            agg["engagement_score"] = ("engagement_score", "mean")

        return hr.groupby("date").agg(**agg).reset_index()

    @cached_property
    def customers(self) -> pd.DataFrame:
        path = self._dir / "customer_data.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()

    # ── Top-level KPIs ────────────────────────────────────────
    @cached_property
    def kpis(self) -> Dict[str, Any]:
        fin  = self.finance
        sal  = self.sales_monthly
        hr_m = self.hr_monthly
        ops  = self.operations

        latest_fin = fin.iloc[-1]
        latest_sal = sal.iloc[-1]
        latest_hr  = hr_m.iloc[-1]
        latest_ops = ops.iloc[-1]

        prior_fin = fin.iloc[-2] if len(fin) > 1 else latest_fin
        prior_sal = sal.iloc[-2] if len(sal) > 1 else latest_sal

        def pct_chg(curr, prev):
            return round((curr - prev) / max(abs(prev), 1) * 100, 1)

        def _get(row, *candidates, default=0):
            for c in candidates:
                if c in row.index:
                    return row[c]
            return default

        total_revenue = round(fin["revenue"].sum(), 0)
        total_profit  = round(fin["net_profit"].sum(), 0)

        # Gross margin — prefer pre-computed column, else derive it
        if "gross_margin_pct" in fin.columns:
            gm = round(fin["gross_margin_pct"].mean(), 1)
        elif "gross_profit" in fin.columns:
            gm = round(fin["gross_profit"].mean() / fin["revenue"].mean() * 100, 1)
        else:
            gm = None

        ops_eff  = _get(latest_ops, "process_efficiency_pct", "server_uptime_pct")
        ops_sla  = _get(latest_ops, "sla_compliance_pct", "customer_satisfaction")
        ops_def  = _get(latest_ops, "defect_rate_pct", "change_failure_rate_pct")
        ops_res  = _get(latest_ops, "avg_resolution_hours", "p1_resolution_hours",
                        "mean_time_to_restore_hrs")
        budget_col = "budget_revenue" if "budget_revenue" in fin.columns else None

        kpi: Dict[str, Any] = {
            # Finance
            "total_revenue":      total_revenue,
            "revenue_delta_pct":  pct_chg(latest_fin["revenue"], prior_fin["revenue"]),
            "total_net_profit":   total_profit,
            "profit_delta_pct":   pct_chg(latest_fin["net_profit"], prior_fin["net_profit"]),
            "gross_margin_pct":   gm,
            "ebitda_latest":      round(float(_get(latest_fin, "ebitda", default=0)), 0),
            # Sales
            "total_units_sold":   int(self.sales["units_sold"].sum()),
            "top_product":        self.sales_by_product.iloc[0]["product"],
            "top_region":         self.sales_by_region.iloc[0]["region"],
            # HR
            "total_headcount":    int(latest_hr["headcount"]),
            "headcount_delta":    int(latest_hr["headcount"] - hr_m.iloc[-2]["headcount"])
                                  if len(hr_m) > 1 else 0,
            "attrition_rate_pct": round(float(latest_hr["attrition_rate"]), 2),
            "new_hires_ytd":      int(self.hr["new_hires"].tail(12).sum()),
            # Operations
            "efficiency_pct":     round(float(ops_eff), 2) if ops_eff else None,
            "sla_compliance_pct": round(float(ops_sla), 2) if ops_sla else None,
            "defect_rate_pct":    round(float(ops_def), 2) if ops_def else None,
            "avg_resolution_hrs": round(float(ops_res), 1) if ops_res else None,
        }

        # Optional extras from new schema
        if budget_col:
            kpi["revenue_vs_budget_pct"] = pct_chg(
                latest_fin["revenue"], latest_fin[budget_col]
            )
        if "arr" in fin.columns:
            kpi["arr_latest"] = round(float(fin["arr"].iloc[-1]), 0)
        if "churn_rate_pct" in fin.columns:
            kpi["churn_rate_pct"] = round(float(fin["churn_rate_pct"].iloc[-1]), 2)
        if not self.customers.empty:
            cust = self.customers
            kpi["total_customers"] = int(cust["total_customers"].iloc[-1])
            if "nps_score" in cust.columns:
                kpi["nps_score"] = round(float(cust["nps_score"].iloc[-1]), 1)
            if "ltv_cac_ratio" in cust.columns:
                kpi["ltv_cac_ratio"] = round(float(cust["ltv_cac_ratio"].iloc[-1]), 2)

        return kpi

    # ── Serialisable summary for AI ───────────────────────────
    def ai_summary(self) -> Dict[str, Any]:
        """Compact dict for LLM prompts — last 3 months, token-efficient."""
        fin_cols = ["date", "revenue", "net_profit", "ebitda"]

        ops_cols = [c for c in ["date", "process_efficiency_pct", "sla_compliance_pct",
                                 "defect_rate_pct", "server_uptime_pct",
                                 "customer_satisfaction"]
                    if c in self.operations.columns]

        hr_cols = [c for c in ["date", "headcount", "attrition_rate", "eNPS"]
                   if c in self.hr_monthly.columns]

        summary: Dict[str, Any] = {
            "kpis":           self.kpis,
            "finance_last_3": self.finance[fin_cols].tail(3).to_dict("records"),
            "operations_last_3": self.operations[ops_cols].tail(3).to_dict("records"),
            "hr_last_3":      self.hr_monthly[hr_cols].tail(3).to_dict("records"),
            "top_3_products": self.sales_by_product.head(3)[["product", "revenue"]].to_dict("records"),
            "top_3_regions":  self.sales_by_region.head(3).to_dict("records"),
        }

        if not self.customers.empty:
            cust_cols = [c for c in ["date", "total_customers", "churn_rate_pct",
                                      "nps_score", "ltv_cac_ratio"]
                         if c in self.customers.columns]
            summary["customers_last_3"] = self.customers[cust_cols].tail(3).to_dict("records")

        return summary

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
