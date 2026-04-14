"""
Pydantic schemas for all data domain entities.
Author: Tuan Tran
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


# ── Sales ─────────────────────────────────────────────────────────────────────
class SalesRecord(BaseModel):
    date: date
    product: str
    region: str
    channel: str                      # Online / Retail / Partner
    units_sold: int = Field(ge=0)
    revenue: float = Field(ge=0)
    cost_of_goods: float = Field(ge=0)
    gross_profit: float
    discount_pct: float = Field(ge=0, le=100)
    sales_rep: str


# ── Finance ───────────────────────────────────────────────────────────────────
class FinancialRecord(BaseModel):
    date: date
    revenue: float
    cogs: float
    gross_profit: float
    operating_expenses: float
    ebitda: float
    net_profit: float
    budget_revenue: float
    budget_opex: float
    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float


# ── HR ────────────────────────────────────────────────────────────────────────
class HRRecord(BaseModel):
    date: date
    department: str
    headcount: int = Field(ge=0)
    new_hires: int = Field(ge=0)
    attritions: int = Field(ge=0)
    attrition_rate_pct: float
    avg_salary: float = Field(ge=0)
    performance_below: int = Field(ge=0)
    performance_meets: int = Field(ge=0)
    performance_exceeds: int = Field(ge=0)
    open_positions: int = Field(ge=0)
    training_hours: float = Field(ge=0)


# ── Operations ────────────────────────────────────────────────────────────────
class OperationsRecord(BaseModel):
    date: date
    process_efficiency_pct: float
    sla_compliance_pct: float
    defect_rate_pct: float
    ticket_volume: int = Field(ge=0)
    avg_resolution_hours: float = Field(ge=0)
    capacity_utilisation_pct: float
    downtime_hours: float = Field(ge=0)
    cost_per_unit: float = Field(ge=0)
