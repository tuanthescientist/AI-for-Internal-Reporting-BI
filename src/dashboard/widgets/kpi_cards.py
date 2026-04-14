"""
KPI Card widgets — compact metric tiles shown at the top of the dashboard.
Author: Tuan Tran
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from src.config.settings import Config


class KPICard(QFrame):
    """
    A single KPI tile displaying:
        • Icon / emoji
        • Title
        • Value (large, bold)
        • Delta badge (▲ green / ▼ red)
        • Sub-label
    """

    def __init__(
        self,
        title: str,
        value: str,
        delta: str = "",
        sub_label: str = "",
        icon: str = "📊",
        positive_delta: Optional[bool] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("kpi_card")
        self.setMinimumWidth(170)
        self.setMaximumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._build_ui(title, value, delta, sub_label, icon, positive_delta)

    def _build_ui(
        self,
        title: str,
        value: str,
        delta: str,
        sub_label: str,
        icon: str,
        positive_delta: Optional[bool],
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        # ── Header row (icon + title) ─────────────────────────
        header = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("Segoe UI Emoji", 15))
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        header.addWidget(icon_lbl)
        header.addWidget(title_lbl)
        header.addStretch()

        # ── Delta badge ───────────────────────────────────────
        if delta:
            if positive_delta is True:
                badge_color = Config.SUCCESS_COLOR
                arrow = "▲"
            elif positive_delta is False:
                badge_color = Config.DANGER_COLOR
                arrow = "▼"
            else:
                badge_color = Config.TEXT_SECONDARY
                arrow = "◆"

            delta_lbl = QLabel(f"{arrow} {delta}")
            delta_lbl.setStyleSheet(
                f"color: {badge_color}; font-size: 10px; font-weight: 600;"
                f"background: rgba(0,0,0,0.3); border-radius: 4px; padding: 1px 5px;"
            )
            header.addWidget(delta_lbl)

        layout.addLayout(header)

        # ── Value ─────────────────────────────────────────────
        value_lbl = QLabel(value)
        value_lbl.setStyleSheet(
            f"color: {Config.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
        )
        layout.addWidget(value_lbl)

        # ── Sub-label ─────────────────────────────────────────
        if sub_label:
            sub_lbl = QLabel(sub_label)
            sub_lbl.setStyleSheet(
                f"color: {Config.TEXT_SECONDARY}; font-size: 10px;"
            )
            layout.addWidget(sub_lbl)

    def update_value(self, value: str, delta: str = "", positive_delta: Optional[bool] = None) -> None:
        """Refresh displayed values (rebuild the widget content)."""
        # Clear existing layout and rebuild — simple approach for mock data
        for i in reversed(range(self.layout().count())):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().deleteLater()


class KPICardsRow(QWidget):
    """Horizontal row of KPI cards."""

    def __init__(self, kpis: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(12)
        self._build(kpis)

    def _build(self, kpis: dict) -> None:
        def fmt_currency(v: float) -> str:
            if v >= 1_000_000:
                return f"${v/1_000_000:.1f}M"
            if v >= 1_000:
                return f"${v/1_000:.0f}K"
            return f"${v:.0f}"

        def fmt_pct(v: float) -> str:
            return f"{v:.1f}%"

        def fmt_int(v: int) -> str:
            return f"{v:,}"

        cards = [
            KPICard(
                title="Total Revenue",
                value=fmt_currency(kpis.get("total_revenue", 0)),
                delta=f"{abs(kpis.get('revenue_delta_pct', 0)):.1f}% MoM",
                sub_label="24-month cumulative",
                icon="💰",
                positive_delta=kpis.get("revenue_delta_pct", 0) >= 0,
            ),
            KPICard(
                title="Net Profit",
                value=fmt_currency(kpis.get("total_net_profit", 0)),
                delta=f"{abs(kpis.get('profit_delta_pct', 0)):.1f}% MoM",
                sub_label="24-month cumulative",
                icon="📈",
                positive_delta=kpis.get("profit_delta_pct", 0) >= 0,
            ),
            KPICard(
                title="Gross Margin",
                value=fmt_pct(kpis.get("gross_margin_pct", 0)),
                sub_label="Average over period",
                icon="🎯",
            ),
            KPICard(
                title="vs Budget",
                value=f"{kpis.get('revenue_vs_budget_pct', 0):+.1f}%",
                sub_label="Revenue vs budget",
                icon="📋",
                positive_delta=kpis.get("revenue_vs_budget_pct", 0) >= 0,
            ),
            KPICard(
                title="Headcount",
                value=fmt_int(kpis.get("total_headcount", 0)),
                delta=f"{kpis.get('headcount_delta', 0):+d} vs prior",
                sub_label="Current total",
                icon="👥",
                positive_delta=kpis.get("headcount_delta", 0) >= 0,
            ),
            KPICard(
                title="Attrition Rate",
                value=fmt_pct(kpis.get("attrition_rate_pct", 0)),
                sub_label="Latest month avg",
                icon="🔄",
                positive_delta=False,
            ),
            KPICard(
                title="Efficiency",
                value=fmt_pct(kpis.get("efficiency_pct", 0)),
                sub_label="Process efficiency",
                icon="⚡",
                positive_delta=kpis.get("efficiency_pct", 0) >= 94,
            ),
            KPICard(
                title="SLA Compliance",
                value=fmt_pct(kpis.get("sla_compliance_pct", 0)),
                sub_label="Latest month",
                icon="✅",
                positive_delta=kpis.get("sla_compliance_pct", 0) >= 95,
            ),
        ]

        for card in cards:
            self._layout.addWidget(card)
