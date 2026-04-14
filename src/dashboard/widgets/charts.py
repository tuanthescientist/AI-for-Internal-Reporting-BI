"""
Matplotlib chart panels embedded in PyQt5 for each business domain.
Author: Tuan Tran
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from src.config.settings import Config

# ── Matplotlib dark theme ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    Config.BG_CARD,
    "axes.facecolor":      Config.BG_CARD,
    "axes.edgecolor":      "#30363D",
    "axes.labelcolor":     Config.TEXT_SECONDARY,
    "xtick.color":         Config.TEXT_SECONDARY,
    "ytick.color":         Config.TEXT_SECONDARY,
    "text.color":          Config.TEXT_PRIMARY,
    "grid.color":          "#21262D",
    "grid.linestyle":      "--",
    "grid.linewidth":      0.6,
    "font.family":         "DejaVu Sans",
    "legend.facecolor":    Config.BG_PANEL,
    "legend.edgecolor":    "#30363D",
    "legend.fontsize":     9,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.titlecolor":     Config.TEXT_PRIMARY,
})

ACCENT   = Config.ACCENT_COLOR
SUCCESS  = Config.SUCCESS_COLOR
WARNING  = Config.WARNING_COLOR
DANGER   = Config.DANGER_COLOR
PALETTE  = [ACCENT, SUCCESS, WARNING, DANGER, "#A371F7", "#F0883E", "#EC6547", "#58A6FF"]


# ─────────────────────────────────────────────────────────────────────────────
class ChartCanvas(FigureCanvas):
    """Base canvas — wraps a Matplotlib Figure for embedding in Qt."""

    def __init__(self, rows: int = 1, cols: int = 1, height: int = 5) -> None:
        self.fig = Figure(figsize=(10, height), dpi=96, tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(height * 80)


# ─────────────────────────────────────────────────────────────────────────────
class SalesChartsPanel(QWidget):
    """2×2 chart grid for the Sales tab."""

    def __init__(self, pipeline, parent=None) -> None:
        super().__init__(parent)
        self._p = pipeline
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = ChartCanvas(rows=2, cols=2, height=9)
        layout.addWidget(self._canvas)
        self._draw()

    def _draw(self) -> None:
        fig = self._canvas.fig
        fig.clear()
        axes = fig.subplots(2, 2)

        sal = self._p.sales_monthly
        by_region = self._p.sales_by_region
        by_product = self._p.sales_by_product.head(6)
        by_channel = self._p.sales_by_channel

        # ── Revenue Trend ──────────────────────────────────────
        ax = axes[0, 0]
        x = range(len(sal))
        ax.fill_between(x, sal["revenue"] / 1e6, alpha=0.15, color=ACCENT)
        ax.plot(x, sal["revenue"] / 1e6, color=ACCENT, linewidth=2, marker="o", markersize=3)
        ax.set_title("Monthly Revenue Trend")
        ax.set_ylabel("Revenue ($M)")
        ax.set_xticks(x[::3])
        ax.set_xticklabels(
            [str(d)[:7] for d in sal["date"].iloc[::3]], rotation=30, ha="right"
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.grid(True, axis="y")

        # ── Revenue by Region (horizontal bar) ────────────────
        ax = axes[0, 1]
        colors = [ACCENT if i == 0 else "#30363D" for i in range(len(by_region))]
        bars = ax.barh(by_region["region"], by_region["revenue"] / 1e6, color=colors)
        ax.set_title("Revenue by Region")
        ax.set_xlabel("Revenue ($M)")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():.1f}M", va="center", fontsize=8,
                    color=Config.TEXT_SECONDARY)
        ax.grid(True, axis="x")

        # ── Top Products ───────────────────────────────────────
        ax = axes[1, 0]
        clrs = [PALETTE[i % len(PALETTE)] for i in range(len(by_product))]
        ax.bar(range(len(by_product)), by_product["revenue"] / 1e6, color=clrs)
        ax.set_xticks(range(len(by_product)))
        ax.set_xticklabels(
            [p.replace(" ", "\n") for p in by_product["product"]],
            fontsize=8,
        )
        ax.set_title("Revenue by Product")
        ax.set_ylabel("Revenue ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.grid(True, axis="y")

        # ── Channel Mix (pie) ──────────────────────────────────
        ax = axes[1, 1]
        pie_colors = PALETTE[:len(by_channel)]
        wedges, texts, autotexts = ax.pie(
            by_channel["revenue"],
            labels=by_channel["channel"],
            autopct="%1.1f%%",
            colors=pie_colors,
            startangle=90,
            wedgeprops={"edgecolor": Config.BG_DARK, "linewidth": 2},
        )
        for at in autotexts:
            at.set_color("#000")
            at.set_fontsize(9)
        ax.set_title("Channel Revenue Mix")

        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
class FinanceChartsPanel(QWidget):
    """Finance tab: P&L, Budget vs Actual, EBITDA, Cash Flow."""

    def __init__(self, pipeline, parent=None) -> None:
        super().__init__(parent)
        self._p = pipeline
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = ChartCanvas(rows=2, cols=2, height=9)
        layout.addWidget(self._canvas)
        self._draw()

    def _draw(self) -> None:
        fig = self._canvas.fig
        fig.clear()
        axes = fig.subplots(2, 2)
        fin = self._p.finance
        x = range(len(fin))
        xlabels = [str(d)[:7] for d in fin["date"]]

        def _col(df, *candidates, default=None):
            for c in candidates:
                if c in df.columns:
                    return df[c]
            return default if default is not None else pd.Series([0] * len(df))

        # ── P&L Waterfall (stacked area) ──────────────────────
        ax = axes[0, 0]
        gross_profit = _col(fin, "gross_profit")
        net_profit   = _col(fin, "net_profit")
        ax.fill_between(x, fin["revenue"] / 1e6, alpha=0.3, color=ACCENT, label="Revenue")
        ax.fill_between(x, gross_profit / 1e6, alpha=0.3, color=SUCCESS, label="Gross Profit")
        ax.fill_between(x, net_profit / 1e6, alpha=0.4, color=WARNING, label="Net Profit")
        ax.plot(x, fin["revenue"] / 1e6, color=ACCENT, linewidth=1.5)
        ax.plot(x, gross_profit / 1e6, color=SUCCESS, linewidth=1.5)
        ax.plot(x, net_profit / 1e6, color=WARNING, linewidth=1.5)
        ax.set_title("P&L Overview")
        ax.set_ylabel("Amount ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Budget vs Actual ───────────────────────────────────
        ax = axes[0, 1]
        budget = _col(fin, "budget_revenue", "budget_ebitda", default=fin["revenue"] * 0.95)
        width = 0.4
        idx = np.arange(len(fin))
        ax.bar(idx - width/2, fin["revenue"] / 1e6, width, color=ACCENT, label="Actual", alpha=0.8)
        ax.bar(idx + width/2, budget / 1e6, width, color="#30363D", label="Budget", alpha=0.8)
        ax.set_title("Revenue: Actual vs Budget")
        ax.set_ylabel("Amount ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(idx[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── EBITDA Trend ───────────────────────────────────────
        ax = axes[1, 0]
        ebitda = _col(fin, "ebitda")
        ebitda_ma = ebitda.rolling(3).mean()
        ax.bar(x, ebitda / 1e6, color=SUCCESS, alpha=0.5, label="EBITDA")
        ax.plot(x, ebitda_ma / 1e6, color=ACCENT, linewidth=2, label="3M MA")
        ax.set_title("EBITDA Trend")
        ax.set_ylabel("EBITDA ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Cash Flow ──────────────────────────────────────────
        ax = axes[1, 1]
        op_cf  = _col(fin, "operating_cash_flow", "net_cash_flow")
        inv_cf = _col(fin, "investing_cash_flow")
        fin_cf = _col(fin, "financing_cash_flow")
        ax.bar(x, op_cf / 1e6,  color=SUCCESS, alpha=0.7, label="Operating")
        if (inv_cf != 0).any():
            ax.bar(x, inv_cf / 1e6, color=DANGER,  alpha=0.7, label="Investing")
        if (fin_cf != 0).any():
            ax.bar(x, fin_cf / 1e6, color=WARNING, alpha=0.7, label="Financing")
        ax.axhline(0, color=Config.TEXT_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_title("Cash Flow / Net Cash Flow")
        ax.set_ylabel("Cash Flow ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
class HRChartsPanel(QWidget):
    """HR tab: Headcount, Attrition, Performance, Training."""

    def __init__(self, pipeline, parent=None) -> None:
        super().__init__(parent)
        self._p = pipeline
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = ChartCanvas(rows=2, cols=2, height=9)
        layout.addWidget(self._canvas)
        self._draw()

    def _draw(self) -> None:
        fig = self._canvas.fig
        fig.clear()
        axes = fig.subplots(2, 2)
        hr     = self._p.hr_monthly
        hr_raw = self._p.hr
        x = range(len(hr))
        xlabels = [str(d)[:7] for d in hr["date"]]

        def _col(df, *candidates, default=None):
            for c in candidates:
                if c in df.columns:
                    return df[c]
            n = len(df)
            return default if default is not None else pd.Series([0] * n, index=df.index)

        # ── Headcount Trend ────────────────────────────────────
        ax = axes[0, 0]
        hc = _col(hr, "headcount")
        hires = _col(hr, "new_hires")
        attritions = _col(hr, "attritions", default=pd.Series([0]*len(hr), index=hr.index))
        ax.fill_between(x, hc, alpha=0.2, color=ACCENT)
        ax.plot(x, hc, color=ACCENT, linewidth=2, label="Headcount")
        ax.fill_between(x, hires, alpha=0.5, color=SUCCESS, label="Hires")
        if attritions.sum() > 0:
            ax.fill_between(x, attritions, alpha=0.5, color=DANGER, label="Attritions")
        ax.set_title("Headcount Trend")
        ax.set_ylabel("No. of Employees")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Attrition Rate ─────────────────────────────────────
        ax = axes[0, 1]
        att_rate = _col(hr, "attrition_rate")
        clrs = [DANGER if v > 4 else SUCCESS for v in att_rate]
        ax.bar(x, att_rate, color=clrs, alpha=0.8)
        ax.axhline(4, color=WARNING, linewidth=1.5, linestyle="--", label="Target ≤4%")
        ax.set_title("Monthly Attrition Rate")
        ax.set_ylabel("Attrition (%)")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Headcount by Department (bar) ──────────────────────
        ax = axes[1, 0]
        hc_col  = "headcount_eop" if "headcount_eop" in hr_raw.columns else "headcount"
        att_col = "attrition_rate_pct" if "attrition_rate_pct" in hr_raw.columns else "attrition_rate"
        if "department" in hr_raw.columns:
            dept_agg = hr_raw.groupby("department")[hc_col].mean().reset_index()
            dept_agg.columns = ["department", "headcount"]
            clrs_d = [PALETTE[i % len(PALETTE)] for i in range(len(dept_agg))]
            ax.bar(dept_agg["department"], dept_agg["headcount"], color=clrs_d, alpha=0.8)
            ax.set_xticks(range(len(dept_agg)))
            ax.set_xticklabels(dept_agg["department"], rotation=30, ha="right")
        ax.set_title("Avg Headcount by Department")
        ax.set_ylabel("Avg Headcount")
        ax.grid(True, axis="y")

        # ── eNPS / Engagement or Training Hours ────────────────
        ax = axes[1, 1]
        if "eNPS" in hr.columns:
            enps = _col(hr, "eNPS")
            ax.plot(x, enps, color=ACCENT, linewidth=2, marker="o", markersize=3, label="eNPS")
            ax.axhline(0, color=Config.TEXT_SECONDARY, linewidth=0.8, linestyle="--")
            ax.set_title("eNPS Trend")
            ax.set_ylabel("eNPS Score")
            ax.legend()
        elif "training_hours" in hr_raw.columns:
            train = hr_raw.groupby("department")["training_hours"].mean().reset_index()
            clrs_t = [PALETTE[i % len(PALETTE)] for i in range(len(train))]
            ax.barh(train["department"], train["training_hours"], color=clrs_t)
            ax.set_title("Avg Monthly Training Hours")
            ax.set_xlabel("Hours")
            ax.axvline(8, color=WARNING, linewidth=1.5, linestyle="--", label="Target 8h")
            ax.legend()
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.grid(True, axis="y")

        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
class OpsChartsPanel(QWidget):
    """Operations tab: Efficiency, SLA, Defects, Tickets."""

    def __init__(self, pipeline, parent=None) -> None:
        super().__init__(parent)
        self._p = pipeline
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = ChartCanvas(rows=2, cols=2, height=9)
        layout.addWidget(self._canvas)
        self._draw()

    def _draw(self) -> None:
        fig = self._canvas.fig
        fig.clear()
        axes = fig.subplots(2, 2)
        ops = self._p.operations
        x = range(len(ops))
        xlabels = [str(d)[:7] for d in ops["date"]]

        def _col(df, *candidates, default=None):
            for c in candidates:
                if c in df.columns:
                    return df[c]
            n = len(df)
            return default if default is not None else pd.Series([0] * n, index=df.index)

        eff_series  = _col(ops, "process_efficiency_pct", "server_uptime_pct", default=pd.Series([95]*len(ops), index=ops.index))
        sla_series  = _col(ops, "sla_compliance_pct", "customer_satisfaction", default=pd.Series([95]*len(ops), index=ops.index))
        def_series  = _col(ops, "defect_rate_pct", "change_failure_rate_pct", default=pd.Series([2]*len(ops), index=ops.index))

        # ── Process Efficiency / Server Uptime ─────────────────
        ax = axes[0, 0]
        clrs = [SUCCESS if v >= 94 else DANGER for v in eff_series]
        ax.bar(x, eff_series, color=clrs, alpha=0.8)
        ax.axhline(94, color=WARNING, linewidth=1.5, linestyle="--", label="Target 94%")
        ax.set_ylim(70, 100)
        eff_title = "Server Uptime (%)" if "server_uptime_pct" in ops.columns else "Process Efficiency (%)"
        ax.set_title(eff_title)
        ax.set_ylabel(eff_title)
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── SLA Compliance / Customer Satisfaction ─────────────
        ax = axes[0, 1]
        ax.fill_between(x, sla_series, 98, alpha=0.2,
                        where=[v < 98 for v in sla_series],
                        color=DANGER, label="Below Target")
        ax.plot(x, sla_series, color=ACCENT, linewidth=2)
        ax.axhline(98, color=WARNING, linewidth=1.5, linestyle="--", label="Target 98%")
        ax.set_ylim(70, 100)
        sla_title = "SLA Compliance (%)" if "sla_compliance_pct" in ops.columns else "Customer Satisfaction"
        ax.set_title(sla_title)
        ax.set_ylabel(sla_title)
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Defect Rate / Change Failure Rate ──────────────────
        ax = axes[1, 0]
        def_ma = def_series.rolling(3).mean()
        ax.bar(x, def_series, color=WARNING, alpha=0.5, label="Rate")
        ax.plot(x, def_ma, color=DANGER, linewidth=2, label="3M MA")
        def_title = "Defect Rate (%)" if "defect_rate_pct" in ops.columns else "Change Failure Rate (%)"
        ax.set_title(def_title)
        ax.set_ylabel(def_title)
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── MTTR / Resolution Time or Deploy Frequency ─────────
        ax = axes[1, 1]
        if "deploy_frequency_month" in ops.columns:
            deploy = ops["deploy_frequency_month"]
            ax.bar(x, deploy, color=ACCENT, alpha=0.7, label="Deploys/Month")
            ax.set_title("Deployment Frequency")
            ax.set_ylabel("Deployments / Month")
        elif "ticket_volume" in ops.columns:
            ax.bar(x, ops["ticket_volume"], color=ACCENT, alpha=0.4, label="Ticket Volume")
            ax.set_title("Ticket Volume")
            ax.set_ylabel("Ticket Count")
        else:
            mttr = _col(ops, "mean_time_to_restore_hrs", "avg_resolution_hours", "p1_resolution_hours")
            ax.plot(x, mttr, color=WARNING, linewidth=2, marker="o", markersize=3, label="MTTR (hrs)")
            ax.set_title("Mean Time to Restore (hrs)")
            ax.set_ylabel("Hours")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        self._canvas.draw()
