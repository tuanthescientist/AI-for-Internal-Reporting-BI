"""
MockDataGenerator — synthesises realistic enterprise datasets.

Generates 24 months (Jan 2024 – Dec 2025) of data across:
  • Sales      → data/mock/sales_data.csv
  • Finance    → data/mock/financial_data.csv
  • HR         → data/mock/hr_data.csv
  • Operations → data/mock/operations_data.csv

Run directly:
    python -m src.data.mock_data_generator

Author: Tuan Tran
"""

from __future__ import annotations

import logging
import random
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config.settings import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)

PRODUCTS = [
    "Enterprise Suite", "Analytics Pro", "DataBridge", "ReportMaster",
    "CloudSync", "InsightHub", "AutoReport", "PulseBoard",
]
REGIONS = ["North", "South", "East", "West", "International"]
CHANNELS = ["Direct", "Online", "Partner", "Reseller"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Operations", "HR", "Finance"]
SALES_REPS = [
    "Alice Nguyen", "Bob Carter", "Carol Lee", "David Kim",
    "Eva Martinez", "Frank Chen", "Grace Park", "Henry Wilson",
]

# Monthly date range: Jan 2024 → Dec 2025
MONTHS: List[date] = pd.date_range("2024-01-01", "2025-12-01", freq="MS").date.tolist()  # type: ignore[attr-defined]


def _seasonal(month: int) -> float:
    """Seasonal multiplier: Q4 boost, Q1 dip."""
    if month in (10, 11, 12):
        return 1.25
    if month in (1, 2):
        return 0.90
    return 1.0


def _growth(idx: int, total: int = 24, cagr: float = 0.15) -> float:
    """Monotonic growth factor over the date range."""
    monthly_rate = (1 + cagr) ** (1 / 12) - 1
    return (1 + monthly_rate) ** idx


# ─────────────────────────────────────────────────────────────────────────────
class MockDataGenerator:
    """Generates and persists mock CSV datasets."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Config.DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────
    def generate_all(self) -> None:
        """Generate all four domain datasets."""
        logger.info("Generating mock datasets → %s", self.output_dir)
        self.sales().to_csv(self.output_dir / "sales_data.csv", index=False)
        self.finance().to_csv(self.output_dir / "financial_data.csv", index=False)
        self.hr().to_csv(self.output_dir / "hr_data.csv", index=False)
        self.operations().to_csv(self.output_dir / "operations_data.csv", index=False)
        logger.info("Mock data generation complete.")

    # ── Sales ─────────────────────────────────────────────────
    def sales(self) -> pd.DataFrame:
        rows = []
        for idx, dt in enumerate(MONTHS):
            base_rev = 180_000
            seasonal = _seasonal(dt.month)
            growth = _growth(idx)
            for product in PRODUCTS:
                for region in REGIONS:
                    channel = RNG.choice(CHANNELS)
                    product_factor = RNG.uniform(0.6, 1.4)
                    region_factor = RNG.uniform(0.5, 1.5)
                    noise = RNG.normal(1.0, 0.08)

                    units = max(1, int(
                        RNG.integers(50, 400) * seasonal * growth * product_factor * region_factor * noise
                    ))
                    price = RNG.uniform(200, 2000)
                    revenue = round(units * price, 2)
                    cogs_rate = RNG.uniform(0.35, 0.55)
                    cog = round(revenue * cogs_rate, 2)
                    discount = round(RNG.uniform(0, 15), 1)

                    rows.append({
                        "date": dt,
                        "product": product,
                        "region": region,
                        "channel": channel,
                        "units_sold": units,
                        "revenue": revenue,
                        "cost_of_goods": cog,
                        "gross_profit": round(revenue - cog, 2),
                        "discount_pct": discount,
                        "sales_rep": RNG.choice(SALES_REPS),
                    })

        return pd.DataFrame(rows)

    # ── Finance ───────────────────────────────────────────────
    def finance(self) -> pd.DataFrame:
        rows = []
        base_rev = 2_400_000
        for idx, dt in enumerate(MONTHS):
            growth = _growth(idx, cagr=0.12)
            seasonal = _seasonal(dt.month)
            noise = RNG.normal(1.0, 0.04)

            revenue = round(base_rev * growth * seasonal * noise, 2)
            cogs = round(revenue * RNG.uniform(0.38, 0.46), 2)
            gross_profit = round(revenue - cogs, 2)
            opex = round(revenue * RNG.uniform(0.28, 0.36), 2)
            ebitda = round(gross_profit - opex, 2)
            net_profit = round(ebitda * RNG.uniform(0.72, 0.85), 2)

            budget_variance = RNG.normal(1.0, 0.06)
            budget_revenue = round(revenue * budget_variance, 2)
            budget_opex = round(opex * RNG.normal(1.0, 0.07), 2)

            rows.append({
                "date": dt,
                "revenue": revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
                "operating_expenses": opex,
                "ebitda": ebitda,
                "net_profit": net_profit,
                "budget_revenue": budget_revenue,
                "budget_opex": budget_opex,
                "operating_cash_flow": round(ebitda * RNG.uniform(0.80, 1.10), 2),
                "investing_cash_flow": round(-RNG.uniform(50_000, 200_000), 2),
                "financing_cash_flow": round(RNG.uniform(-100_000, 50_000), 2),
            })

        return pd.DataFrame(rows)

    # ── HR ────────────────────────────────────────────────────
    def hr(self) -> pd.DataFrame:
        rows = []
        base_hc = {dept: RNG.integers(20, 120) for dept in DEPARTMENTS}
        for idx, dt in enumerate(MONTHS):
            for dept in DEPARTMENTS:
                hc = base_hc[dept]
                hires = int(RNG.integers(0, max(1, hc // 10)))
                attritions = int(RNG.integers(0, max(1, hc // 20)))
                base_hc[dept] = max(10, hc + hires - attritions)

                total = hc
                below = max(0, int(total * RNG.uniform(0.07, 0.13)))
                exceeds = max(0, int(total * RNG.uniform(0.20, 0.30)))
                meets = max(0, total - below - exceeds)

                rows.append({
                    "date": dt,
                    "department": dept,
                    "headcount": hc,
                    "new_hires": hires,
                    "attritions": attritions,
                    "attrition_rate_pct": round(attritions / max(1, hc) * 100, 2),
                    "avg_salary": round(RNG.uniform(55_000, 150_000), 0),
                    "performance_below": below,
                    "performance_meets": meets,
                    "performance_exceeds": exceeds,
                    "open_positions": int(RNG.integers(0, 10)),
                    "training_hours": round(RNG.uniform(2, 20), 1),
                })

        return pd.DataFrame(rows)

    # ── Operations ────────────────────────────────────────────
    def operations(self) -> pd.DataFrame:
        rows = []
        for idx, dt in enumerate(MONTHS):
            # Introduce a realistic anomaly around month 14 (Feb 2025)
            anomaly_factor = 0.75 if idx == 14 else 1.0

            rows.append({
                "date": dt,
                "process_efficiency_pct": round(
                    min(99.9, RNG.normal(94.5, 1.5) * anomaly_factor), 2
                ),
                "sla_compliance_pct": round(
                    min(99.9, RNG.normal(97.2, 1.2) * anomaly_factor), 2
                ),
                "defect_rate_pct": round(
                    max(0.1, RNG.normal(1.8, 0.4) / anomaly_factor), 2
                ),
                "ticket_volume": int(
                    RNG.integers(300, 900) / anomaly_factor
                ),
                "avg_resolution_hours": round(
                    RNG.uniform(4, 24) / anomaly_factor, 1
                ),
                "capacity_utilisation_pct": round(
                    min(99.0, RNG.normal(78, 6)), 2
                ),
                "downtime_hours": round(
                    max(0, RNG.normal(2, 1.5)) / anomaly_factor, 2
                ),
                "cost_per_unit": round(RNG.uniform(12, 35), 2),
            })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    MockDataGenerator().generate_all()
