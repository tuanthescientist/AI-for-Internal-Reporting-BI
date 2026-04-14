"""
MockDataGenerator — synthesises professional-grade enterprise datasets.

Generates 36 months (Jan 2023 - Dec 2025) of rich, realistic data across:
  • Sales      → data/mock/sales_data.csv
  • Finance    → data/mock/financial_data.csv
  • HR         → data/mock/hr_data.csv
  • Operations → data/mock/operations_data.csv
  • Customers  → data/mock/customer_data.csv

Design principles:
  - Realistic seasonality (Q4 peak, Jan dip, summer uptick)
  - Compound annual growth rates per domain
  - Correlated KPIs (e.g. headcount growth tracks revenue)
  - Deliberate anomalies seeded at realistic crisis points
  - Product pricing tiers with distinct margin profiles
  - Geographic market share distribution (Vietnam + SEA + Global)

Run directly:
    python -m src.data.mock_data_generator

Author: Tuan Tran
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config.settings import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
SEED = 2024
RNG  = np.random.default_rng(SEED)

# ── Product catalogue with realistic pricing & margin tiers ───────────────────
PRODUCTS: List[Dict] = [
    {"name": "Enterprise Suite",  "tier": "Enterprise", "base_price": 4_800, "cogs_rate": 0.28, "base_units": 35},
    {"name": "Analytics Pro",     "tier": "Enterprise", "base_price": 3_200, "cogs_rate": 0.30, "base_units": 55},
    {"name": "DataBridge API",    "tier": "Platform",   "base_price": 1_500, "cogs_rate": 0.35, "base_units": 120},
    {"name": "ReportMaster",      "tier": "Platform",   "base_price": 1_200, "cogs_rate": 0.38, "base_units": 140},
    {"name": "CloudSync",         "tier": "Cloud",      "base_price":   800, "cogs_rate": 0.42, "base_units": 200},
    {"name": "InsightHub",        "tier": "Cloud",      "base_price":   750, "cogs_rate": 0.40, "base_units": 180},
    {"name": "AutoReport Lite",   "tier": "SMB",        "base_price":   290, "cogs_rate": 0.50, "base_units": 450},
    {"name": "PulseBoard Starter","tier": "SMB",        "base_price":   220, "cogs_rate": 0.52, "base_units": 520},
    {"name": "AI Insights Add-on","tier": "Add-on",     "base_price":   600, "cogs_rate": 0.22, "base_units": 90},
    {"name": "Security Module",   "tier": "Add-on",     "base_price":   480, "cogs_rate": 0.25, "base_units": 75},
]

# ── Geographic markets with relative weightings ───────────────────────────────
MARKETS: List[Dict] = [
    {"region": "Vietnam",        "country": "VN", "weight": 0.28, "growth_premium": 0.08},
    {"region": "Southeast Asia", "country": "SEA","weight": 0.25, "growth_premium": 0.06},
    {"region": "East Asia",      "country": "EA", "weight": 0.20, "growth_premium": 0.04},
    {"region": "North America",  "country": "US", "weight": 0.15, "growth_premium": 0.02},
    {"region": "Europe",         "country": "EU", "weight": 0.08, "growth_premium": 0.01},
    {"region": "Rest of World",  "country": "ROW","weight": 0.04, "growth_premium": 0.00},
]

CHANNELS    = ["Direct Sales", "Online / SaaS", "Partner / VAR", "Reseller", "eCommerce"]
CHANNEL_W   = [0.30, 0.35, 0.18, 0.12, 0.05]   # channel probability weights

DEPARTMENTS: List[Dict] = [
    {"name": "Engineering",  "base_hc": 85,  "salary_min": 80_000, "salary_max": 160_000},
    {"name": "Sales",        "base_hc": 55,  "salary_min": 60_000, "salary_max": 130_000},
    {"name": "Marketing",    "base_hc": 30,  "salary_min": 58_000, "salary_max": 110_000},
    {"name": "Operations",   "base_hc": 45,  "salary_min": 50_000, "salary_max": 100_000},
    {"name": "HR & People",  "base_hc": 18,  "salary_min": 52_000, "salary_max": 95_000},
    {"name": "Finance",      "base_hc": 22,  "salary_min": 65_000, "salary_max": 120_000},
    {"name": "Data & AI",    "base_hc": 28,  "salary_min": 90_000, "salary_max": 175_000},
    {"name": "Product",      "base_hc": 20,  "salary_min": 85_000, "salary_max": 155_000},
]

SALES_REPS = [
    "Nguyen Thi Lan", "Tran Minh Duc", "Le Thi Hoa", "Pham Van Khanh",
    "Hoang Thi Mai",  "Vu Duc Anh",    "Do Thi Phuong", "Bui Van Tuan",
    "Sarah Johnson",  "Michael Chen",  "Emma Davis",    "James Wilson",
]

CUSTOMER_SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]

# ── 36-month date range: Jan 2023 → Dec 2025 ────────────────────────────────
MONTHS: List[date] = pd.date_range("2023-01-01", "2025-12-01", freq="MS").date.tolist()  # type: ignore[attr-defined]
N_MONTHS = len(MONTHS)   # 36


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _seasonal(month: int) -> float:
    """
    Realistic B2B SaaS seasonality:
      Q4  (Oct-Dec): +30% — budget flush, year-end deals
      Q1  (Jan-Feb): -15% — post-holiday slow start
      Aug:           -5%  — summer slowdown
      Others: baseline
    """
    profile = {1: 0.82, 2: 0.88, 3: 1.00, 4: 1.05, 5: 1.08, 6: 1.10,
               7: 1.05, 8: 0.95, 9: 1.12, 10: 1.20, 11: 1.25, 12: 1.30}
    return profile.get(month, 1.0)


def _growth(idx: int, cagr: float = 0.18) -> float:
    """Compound monthly growth factor."""
    monthly_rate = (1 + cagr) ** (1 / 12) - 1
    return (1 + monthly_rate) ** idx


def _noise(sigma: float = 0.05) -> float:
    return float(RNG.normal(1.0, sigma))


# ─────────────────────────────────────────────────────────────────────────────
class MockDataGenerator:
    """
    Generates highly realistic, internally consistent enterprise datasets.
    All domains are cross-correlated: revenue growth drives hiring, headcount
    drives OpEx, ops efficiency influences margins.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Config.DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────
    def generate_all(self) -> None:
        logger.info("Generating professional mock datasets → %s", self.output_dir)
        fin_df  = self.finance()
        hr_df   = self.hr()
        ops_df  = self.operations()
        sal_df  = self.sales(fin_df)
        cust_df = self.customers(sal_df)

        fin_df.to_csv(self.output_dir  / "financial_data.csv",  index=False)
        hr_df.to_csv(self.output_dir   / "hr_data.csv",         index=False)
        ops_df.to_csv(self.output_dir  / "operations_data.csv", index=False)
        sal_df.to_csv(self.output_dir  / "sales_data.csv",      index=False)
        cust_df.to_csv(self.output_dir / "customer_data.csv",   index=False)
        logger.info("Done. 5 datasets written.")
        for f in self.output_dir.iterdir():
            if f.suffix == ".csv":
                rows = sum(1 for _ in open(f)) - 1
                logger.info("  %-28s  %6d rows", f.name, rows)

    # ── Finance (anchor dataset — others are derived) ────────
    def finance(self) -> pd.DataFrame:
        """
        Monthly P&L, budget vs actual, cash flow, capex, working capital.
        Seeds:
          - M17 (May 2024): revenue miss due to simulated market disruption
          - M29 (May 2025): record quarter — product launch
        """
        rows = []
        base_rev = 3_200_000   # Jan 2023 starting monthly revenue

        for idx, dt in enumerate(MONTHS):
            g  = _growth(idx, cagr=0.20)
            s  = _seasonal(dt.month)
            n  = _noise(0.035)

            # Deliberate events
            event_factor = 1.0
            if idx == 17:   event_factor = 0.72   # May 2024: supply-chain crisis
            if idx == 29:   event_factor = 1.35   # May 2025: new AI product launch
            if idx in (5, 6): event_factor = 0.95 # Jun-Jul 2023: summer churn

            rev = round(base_rev * g * s * n * event_factor, 0)

            cogs_rate = RNG.uniform(0.34, 0.42)
            cogs      = round(rev * cogs_rate, 0)
            gross     = rev - cogs

            # OpEx scales with revenue but has natural lag
            opex_rate = RNG.uniform(0.30, 0.38) * (1 - 0.002 * idx)  # efficiency improves
            opex      = round(rev * opex_rate, 0)

            rd_expense    = round(rev * RNG.uniform(0.06, 0.10), 0)
            sm_expense    = round(rev * RNG.uniform(0.10, 0.16), 0)
            ga_expense    = round(rev * RNG.uniform(0.05, 0.08), 0)
            total_opex    = round(rd_expense + sm_expense + ga_expense, 0)

            ebitda     = round(gross - total_opex, 0)
            depr_amort = round(rev * 0.02, 0)
            ebit       = round(ebitda - depr_amort, 0)
            interest   = round(rev * RNG.uniform(0.005, 0.012), 0)
            tax_rate   = RNG.uniform(0.18, 0.22)
            ebt        = ebit - interest
            net_profit = round(ebt * (1 - tax_rate), 0)

            # Budget (set at start of year with ±8% uncertainty)
            bud_base       = base_rev * _growth(idx - (idx % 12), cagr=0.15) * s
            budget_rev     = round(bud_base * RNG.normal(1.0, 0.08), 0)
            budget_opex    = round(total_opex * RNG.normal(1.0, 0.07), 0)
            budget_ebitda  = round(budget_rev * 0.18, 0)

            # Cash flow
            op_cf  = round(ebitda * RNG.uniform(0.82, 1.08), 0)
            inv_cf = round(-RNG.uniform(80_000, 350_000), 0)
            fin_cf = round(RNG.uniform(-150_000, 80_000), 0)

            # Balance sheet items
            arr    = round(rev * 12 * RNG.uniform(0.92, 1.05), 0)   # ARR proxy
            mrr    = round(rev * RNG.uniform(0.92, 1.05), 0)
            churn  = round(RNG.uniform(0.8, 2.5), 2)

            rows.append({
                "date":               dt,
                "revenue":            rev,
                "cogs":               cogs,
                "gross_profit":       gross,
                "gross_margin_pct":   round(gross / rev * 100, 2),
                "rd_expense":         rd_expense,
                "sm_expense":         sm_expense,
                "ga_expense":         ga_expense,
                "total_opex":         total_opex,
                "ebitda":             ebitda,
                "ebitda_margin_pct":  round(ebitda / rev * 100, 2),
                "ebit":               ebit,
                "net_profit":         net_profit,
                "net_margin_pct":     round(net_profit / rev * 100, 2),
                "budget_revenue":     budget_rev,
                "budget_opex":        budget_opex,
                "budget_ebitda":      budget_ebitda,
                "revenue_vs_budget_pct": round((rev - budget_rev) / max(1, budget_rev) * 100, 2),
                "operating_cash_flow":   op_cf,
                "investing_cash_flow":   inv_cf,
                "financing_cash_flow":   fin_cf,
                "net_cash_flow":         op_cf + inv_cf + fin_cf,
                "arr":                arr,
                "mrr":                mrr,
                "churn_rate_pct":     churn,
            })

        return pd.DataFrame(rows)

    # ── Sales (derived from Finance to ensure consistency) ────
    def sales(self, finance_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Transactional sales records.  Revenue totals reconcile (+/-5%) to
        the Finance P&L figures for internal consistency.
        """
        rows = []
        fin_rev_by_month: Dict = {}
        if finance_df is not None:
            for _, row in finance_df.iterrows():
                fin_rev_by_month[str(row["date"])[:7]] = float(row["revenue"])

        for idx, dt in enumerate(MONTHS):
            target_rev  = fin_rev_by_month.get(str(dt)[:7], 3_000_000 * _growth(idx))
            total_rev   = 0.0

            for mkt in MARKETS:
                mkt_rev_target = target_rev * mkt["weight"]
                for prod in PRODUCTS:
                    g  = _growth(idx, cagr=0.18 + mkt["growth_premium"])
                    s  = _seasonal(dt.month)
                    pf = RNG.uniform(0.75, 1.30)
                    n  = _noise(0.07)

                    units = max(1, int(prod["base_units"] * g * s * pf * n
                                       * mkt["weight"] * 6))
                    price   = prod["base_price"] * RNG.uniform(0.90, 1.15)
                    revenue = round(units * price, 2)
                    cog     = round(revenue * prod["cogs_rate"] * RNG.uniform(0.95, 1.05), 2)
                    discount = round(RNG.uniform(0, 18) if prod["tier"] in ("Enterprise","Platform")
                                     else RNG.uniform(0, 8), 1)
                    channel  = RNG.choice(CHANNELS, p=CHANNEL_W)
                    segment  = RNG.choice(
                        CUSTOMER_SEGMENTS,
                        p=[0.30, 0.30, 0.28, 0.12] if prod["tier"] in ("Enterprise","Platform")
                          else [0.05, 0.25, 0.45, 0.25]
                    )
                    rep = RNG.choice(SALES_REPS)

                    total_rev += revenue
                    rows.append({
                        "date":           dt,
                        "year":           dt.year,
                        "month":          dt.month,
                        "quarter":        f"Q{(dt.month-1)//3+1}",
                        "product":        prod["name"],
                        "product_tier":   prod["tier"],
                        "region":         mkt["region"],
                        "country_code":   mkt["country"],
                        "channel":        channel,
                        "customer_segment": segment,
                        "units_sold":     units,
                        "unit_price":     round(price, 2),
                        "gross_revenue":  revenue,
                        "discount_pct":   discount,
                        "net_revenue":    round(revenue * (1 - discount/100), 2),
                        "cost_of_goods":  cog,
                        "gross_profit":   round(revenue - cog, 2),
                        "gross_margin_pct": round((revenue - cog) / max(1, revenue) * 100, 1),
                        "sales_rep":      rep,
                        "is_new_customer": bool(RNG.uniform() < 0.28),
                        "deal_type":      RNG.choice(["New Business","Expansion","Renewal"],
                                                     p=[0.28, 0.30, 0.42]),
                    })

        return pd.DataFrame(rows)

    # ── HR ────────────────────────────────────────────────────
    def hr(self) -> pd.DataFrame:
        """
        Monthly workforce metrics per department.
        Headcount grows with revenue (talent following business growth).
        Q3 typically sees higher attrition (summer moves).
        """
        rows = []
        hc: Dict[str, int] = {d["name"]: d["base_hc"] for d in DEPARTMENTS}

        for idx, dt in enumerate(MONTHS):
            g = _growth(idx, cagr=0.12)

            # Reorg event: M16 (Apr 2024) — restructuring reduced head by ~8%
            reorg = (idx == 16)

            for dept in DEPARTMENTS:
                target_hc   = int(dept["base_hc"] * g)
                current_hc  = hc[dept["name"]]

                # Attrition is higher in Q3 and during reorg
                base_attr_rate = 0.018  # monthly ~2%
                if dt.month in (7, 8):
                    base_attr_rate = 0.030
                if reorg:
                    base_attr_rate = 0.055

                attritions  = max(0, int(current_hc * base_attr_rate * RNG.uniform(0.5, 1.5)))
                gap         = target_hc - (current_hc - attritions)
                hires       = max(0, int(gap + RNG.integers(-2, 4))) if gap > 0 else 0
                new_hc      = max(dept["base_hc"] // 2, current_hc - attritions + hires)
                hc[dept["name"]] = new_hc

                total  = current_hc
                below  = max(0, int(total * RNG.uniform(0.06, 0.12)))
                exc    = max(0, int(total * RNG.uniform(0.22, 0.32)))
                meets  = max(0, total - below - exc)

                avg_sal = dept["salary_min"] + (dept["salary_max"] - dept["salary_min"]) * (
                    RNG.uniform(0.4, 0.85) * (1 + 0.004 * idx)  # annual raises
                )
                eNPS = int(RNG.integers(20, 75))
                engagement = round(RNG.uniform(62, 88), 1)

                rows.append({
                    "date":              dt,
                    "year":              dt.year,
                    "month":             dt.month,
                    "quarter":           f"Q{(dt.month-1)//3+1}",
                    "department":        dept["name"],
                    "headcount":         current_hc,
                    "headcount_eop":     new_hc,
                    "new_hires":         hires,
                    "attritions":        attritions,
                    "attrition_rate_pct": round(attritions / max(1, current_hc) * 100, 2),
                    "internal_transfers": int(RNG.integers(0, 3)),
                    "open_positions":    max(0, int(target_hc - new_hc + RNG.integers(0, 5))),
                    "avg_salary_usd":    round(avg_sal, 0),
                    "total_salary_cost": round(avg_sal * current_hc / 12, 0),
                    "performance_below": below,
                    "performance_meets": meets,
                    "performance_exceeds": exc,
                    "training_hours":    round(RNG.uniform(3, 24), 1),
                    "eNPS":              eNPS,
                    "engagement_score":  engagement,
                    "sick_days_avg":     round(RNG.uniform(0.5, 3.0), 1),
                    "overtime_hrs_avg":  round(RNG.uniform(0, 12), 1),
                })

        return pd.DataFrame(rows)

    # ── Operations ────────────────────────────────────────────
    def operations(self) -> pd.DataFrame:
        """
        Platform & delivery KPIs.
        Events:
          M14 (Feb 2024): infrastructure outage → SLA breach
          M17 (May 2024): same market disruption that hit revenue
          M28 (Apr 2025): major platform upgrade → temporary efficiency drop
        """
        rows = []

        for idx, dt in enumerate(MONTHS):
            # Baseline efficiency improves slowly with maturity
            base_eff = 93.0 + idx * 0.08

            event     = 1.0
            crisis    = False
            if idx == 14:  event = 0.72;  crisis = True   # infra outage
            if idx == 17:  event = 0.85;  crisis = True   # market disruption
            if idx == 28:  event = 0.90;  crisis = True   # platform upgrade

            eff  = round(min(99.5, base_eff * event * _noise(0.012)), 2)
            sla  = round(min(99.9, 97.0 * event * _noise(0.010)), 2)
            defect = round(max(0.1, (2.5 - idx * 0.03) / event * _noise(0.08)), 2)

            rows.append({
                "date":                      dt,
                "year":                      dt.year,
                "month":                     dt.month,
                "quarter":                   f"Q{(dt.month-1)//3+1}",
                "process_efficiency_pct":    eff,
                "sla_compliance_pct":        sla,
                "defect_rate_pct":           defect,
                "ticket_volume":             int(RNG.integers(400, 1_100) / event),
                "critical_tickets":          int(RNG.integers(5, 40) / event),
                "avg_resolution_hours":      round(RNG.uniform(3, 18) / event, 1),
                "p1_resolution_hours":       round(RNG.uniform(1, 6) / event, 1),
                "capacity_utilisation_pct":  round(min(98, RNG.normal(76, 5)), 2),
                "server_uptime_pct":         round(min(99.99, 99.5 * event * _noise(0.003)), 3),
                "deploy_frequency_month":    int(RNG.integers(8, 35)),
                "change_failure_rate_pct":   round(max(0, RNG.normal(4, 1.5) / event), 2),
                "mean_time_to_restore_hrs":  round(RNG.uniform(0.5, 8) / event, 2),
                "downtime_hours":            round(max(0, RNG.normal(1.5, 1.0) / event), 2),
                "cost_per_ticket":           round(RNG.uniform(18, 55), 2),
                "automation_rate_pct":       round(min(95, 35 + idx * 1.2 + RNG.normal(0, 3)), 1),
                "customer_satisfaction":     round(min(5.0, (3.8 + idx * 0.02) * event * _noise(0.04)), 2),
                "is_crisis_month":           crisis,
            })

        return pd.DataFrame(rows)

    # ── Customers ─────────────────────────────────────────────
    def customers(self, sales_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Monthly cohort-level customer metrics: acquisition, retention, LTV, NPS."""
        rows = []
        n_customers = 820  # starting base Jan 2023

        for idx, dt in enumerate(MONTHS):
            g           = _growth(idx, cagr=0.15)
            n_target    = int(n_customers * g)
            n_new       = max(0, int(RNG.normal(n_target * 0.045, n_target * 0.008)))
            churn_rate  = round(RNG.uniform(0.012, 0.030), 4)
            churned     = max(0, int(n_customers * churn_rate))
            n_customers = max(100, n_customers + n_new - churned)

            arpu = round((3_200_000 * _growth(idx) * _seasonal(dt.month)) / max(1, n_customers), 2)

            rows.append({
                "date":                  dt,
                "year":                  dt.year,
                "month":                 dt.month,
                "quarter":               f"Q{(dt.month-1)//3+1}",
                "total_customers":       n_customers,
                "new_customers":         n_new,
                "churned_customers":     churned,
                "net_customer_growth":   n_new - churned,
                "churn_rate_pct":        round(churn_rate * 100, 2),
                "retention_rate_pct":    round((1 - churn_rate) * 100, 2),
                "arpu":                  arpu,
                "nps_score":             int(RNG.integers(28, 72)),
                "csat_score":            round(RNG.uniform(3.6, 4.8), 2),
                "ltv_estimate":          round(arpu * 12 / max(churn_rate * 12, 0.01), 0),
                "cac":                   round(RNG.uniform(800, 3_200) * (1 - 0.005 * idx), 0),
                "ltv_cac_ratio":         round((arpu * 12 / max(churn_rate * 12, 0.01)) /
                                               max(1, RNG.uniform(800, 3_200)), 2),
                "trial_conversions":     int(RNG.integers(20, 120)),
                "expansion_revenue_pct": round(RNG.uniform(18, 38), 1),
            })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    MockDataGenerator().generate_all()
