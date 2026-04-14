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

        # ── P&L Waterfall (stacked area) ──────────────────────
        ax = axes[0, 0]
        ax.fill_between(x, fin["revenue"] / 1e6, alpha=0.3, color=ACCENT, label="Revenue")
        ax.fill_between(x, fin["gross_profit"] / 1e6, alpha=0.3, color=SUCCESS, label="Gross Profit")
        ax.fill_between(x, fin["net_profit"] / 1e6, alpha=0.4, color=WARNING, label="Net Profit")
        ax.plot(x, fin["revenue"] / 1e6, color=ACCENT, linewidth=1.5)
        ax.plot(x, fin["gross_profit"] / 1e6, color=SUCCESS, linewidth=1.5)
        ax.plot(x, fin["net_profit"] / 1e6, color=WARNING, linewidth=1.5)
        ax.set_title("P&L Overview")
        ax.set_ylabel("Amount ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Budget vs Actual ───────────────────────────────────
        ax = axes[0, 1]
        width = 0.4
        idx = np.arange(len(fin))
        ax.bar(idx - width/2, fin["revenue"] / 1e6, width, color=ACCENT, label="Actual", alpha=0.8)
        ax.bar(idx + width/2, fin["budget_revenue"] / 1e6, width, color="#30363D",
               label="Budget", alpha=0.8)
        ax.set_title("Revenue: Actual vs Budget")
        ax.set_ylabel("Amount ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_xticks(idx[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── EBITDA Trend ───────────────────────────────────────
        ax = axes[1, 0]
        ebitda_ma = fin["ebitda"].rolling(3).mean()
        ax.bar(x, fin["ebitda"] / 1e6, color=SUCCESS, alpha=0.5, label="EBITDA")
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
        ax.bar(x, fin["operating_cash_flow"] / 1e6, color=SUCCESS, alpha=0.7, label="Operating")
        ax.bar(x, fin["investing_cash_flow"] / 1e6, color=DANGER, alpha=0.7, label="Investing")
        ax.bar(x, fin["financing_cash_flow"] / 1e6, color=WARNING, alpha=0.7, label="Financing")
        ax.axhline(0, color=Config.TEXT_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_title("Cash Flow Components")
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
        hr = self._p.hr_monthly
        hr_dept = self._p.hr.groupby("department").agg(
            headcount=("headcount", "mean"),
            attrition=("attrition_rate_pct", "mean"),
        ).reset_index()
        x = range(len(hr))
        xlabels = [str(d)[:7] for d in hr["date"]]

        # ── Headcount Trend ────────────────────────────────────
        ax = axes[0, 0]
        ax.fill_between(x, hr["headcount"], alpha=0.2, color=ACCENT)
        ax.plot(x, hr["headcount"], color=ACCENT, linewidth=2)
        ax.fill_between(x, hr["new_hires"], alpha=0.5, color=SUCCESS, label="Hires")
        ax.fill_between(x, hr["attritions"], alpha=0.5, color=DANGER, label="Attritions")
        ax.set_title("Headcount Trend")
        ax.set_ylabel("No. of Employees")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Attrition Rate ─────────────────────────────────────
        ax = axes[0, 1]
        clrs = [DANGER if v > 4 else SUCCESS for v in hr["attrition_rate"]]
        ax.bar(x, hr["attrition_rate"], color=clrs, alpha=0.8)
        ax.axhline(4, color=WARNING, linewidth=1.5, linestyle="--", label="Target ≤4%")
        ax.set_title("Monthly Attrition Rate")
        ax.set_ylabel("Attrition (%)")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Performance Distribution (stacked bar by dept) ─────
        ax = axes[1, 0]
        perf = self._p.hr.groupby("department").agg(
            below=("performance_below", "sum"),
            meets=("performance_meets", "sum"),
            exceeds=("performance_exceeds", "sum"),
        ).reset_index()
        depts = perf["department"]
        ax.bar(depts, perf["below"], label="Below", color=DANGER, alpha=0.8)
        ax.bar(depts, perf["meets"], bottom=perf["below"], label="Meets", color=WARNING, alpha=0.8)
        ax.bar(depts, perf["exceeds"], bottom=perf["below"] + perf["meets"],
               label="Exceeds", color=SUCCESS, alpha=0.8)
        ax.set_title("Performance Distribution by Dept")
        ax.set_ylabel("Employee Count")
        ax.set_xticklabels(depts, rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Avg Training Hours by Dept ─────────────────────────
        ax = axes[1, 1]
        train = self._p.hr.groupby("department")["training_hours"].mean().reset_index()
        clrs = [PALETTE[i % len(PALETTE)] for i in range(len(train))]
        ax.barh(train["department"], train["training_hours"], color=clrs)
        ax.set_title("Avg Monthly Training Hours")
        ax.set_xlabel("Hours")
        ax.axvline(8, color=WARNING, linewidth=1.5, linestyle="--", label="Target 8h")
        ax.legend()
        ax.grid(True, axis="x")

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

        # ── Process Efficiency ─────────────────────────────────
        ax = axes[0, 0]
        clrs = [SUCCESS if v >= 94 else DANGER for v in ops["process_efficiency_pct"]]
        ax.bar(x, ops["process_efficiency_pct"], color=clrs, alpha=0.8)
        ax.axhline(94, color=WARNING, linewidth=1.5, linestyle="--", label="Target 94%")
        ax.set_ylim(70, 100)
        ax.set_title("Process Efficiency (%)")
        ax.set_ylabel("Efficiency (%)")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── SLA Compliance ─────────────────────────────────────
        ax = axes[0, 1]
        ax.fill_between(x, ops["sla_compliance_pct"], 98, alpha=0.2,
                        where=[v < 98 for v in ops["sla_compliance_pct"]],
                        color=DANGER, label="Below SLA")
        ax.plot(x, ops["sla_compliance_pct"], color=ACCENT, linewidth=2)
        ax.axhline(98, color=WARNING, linewidth=1.5, linestyle="--", label="SLA Target 98%")
        ax.set_ylim(70, 100)
        ax.set_title("SLA Compliance (%)")
        ax.set_ylabel("SLA (%)")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Defect Rate ────────────────────────────────────────
        ax = axes[1, 0]
        defect_ma = ops["defect_rate_pct"].rolling(3).mean()
        ax.bar(x, ops["defect_rate_pct"], color=WARNING, alpha=0.5, label="Defect Rate")
        ax.plot(x, defect_ma, color=DANGER, linewidth=2, label="3M MA")
        ax.set_title("Defect Rate (%)")
        ax.set_ylabel("Defect Rate (%)")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

        # ── Ticket Volume & Capacity ───────────────────────────
        ax = axes[1, 1]
        ax2 = ax.twinx()
        ax.bar(x, ops["ticket_volume"], color=ACCENT, alpha=0.4, label="Ticket Volume")
        ax2.plot(x, ops["capacity_utilisation_pct"], color=SUCCESS, linewidth=2,
                 label="Capacity Util.")
        ax.set_title("Ticket Volume & Capacity Utilisation")
        ax.set_ylabel("Ticket Count")
        ax2.set_ylabel("Capacity (%)", color=SUCCESS)
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(xlabels[::4], rotation=30, ha="right")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax.grid(True, axis="y")

        self._canvas.draw()
