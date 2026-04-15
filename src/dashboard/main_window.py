"""
MainWindow — the primary PyQt5 application window.

Layout:
┌────────────────────────────────────────────────────────────────┐
│  HEADER  (Logo · Title · User · Date)                         │
├────────────────────────────────────────────────────────────────┤
│  KPI CARDS ROW (8 metrics)                                    │
├─────────────────────────────────┬──────────────────────────────┤
│  TAB CONTENT                    │  AI ASSISTANT PANEL          │
│  [Sales][Finance][HR][Ops]      │  Streaming Qwen 3 chat       │
│  Charts (2×2 matplotlib grid)   │  Report / Anomaly / Insights │
├─────────────────────────────────┴──────────────────────────────┤
│  DATA TABLE  (sortable / filterable / exportable)              │
├────────────────────────────────────────────────────────────────┤
│  STATUS BAR                                                    │
└────────────────────────────────────────────────────────────────┘

Author: Tuan Tran
"""

from __future__ import annotations

from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QFontDatabase, QIcon, QPixmap, QImage, QColor
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QScrollArea, QSizePolicy,
    QSplitter, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from src.config.settings import Config
from src.data.data_pipeline import DataPipeline
from src.ai.qwen_client import QwenAIClient
from src.dashboard.styles import MAIN_STYLE, KPI_CARD_STYLE, AI_PANEL_STYLE
from src.dashboard.widgets.kpi_cards import KPICardsRow
from src.dashboard.widgets.charts import (
    SalesChartsPanel, FinanceChartsPanel, HRChartsPanel, OpsChartsPanel,
)
from src.dashboard.widgets.ai_panel import AIAssistantPanel
from src.dashboard.widgets.data_table import DataTableWidget


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = DataPipeline()
        self._ai_client = QwenAIClient()
        self._setup_window()
        self._build_ui()
        self._apply_styles()
        self._start_clock()

    # ── Window setup ──────────────────────────────────────────
    def _setup_window(self) -> None:
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.setMinimumSize(Config.WINDOW_MIN_WIDTH, Config.WINDOW_MIN_HEIGHT)
        self.resize(Config.WINDOW_MIN_WIDTH + 200, Config.WINDOW_MIN_HEIGHT + 100)
        # Set taskbar / title-bar icon
        logo_px = self._load_logo(32)
        if logo_px:
            self.setWindowIcon(QIcon(logo_px))

    # ── Logo loader ───────────────────────────────────────────
    @staticmethod
    def _load_logo(size: int) -> "QPixmap | None":
        """
        Load 'AI Forecast logo_theme.png', scale to *size* px, and
        convert white pixels → transparent, dark pixels → white so the
        logo looks correct on the dark dashboard background.
        """
        logo_path = Config.ROOT_DIR / "AI Forecast logo_theme.png"
        if not logo_path.exists():
            return None
        pm = QPixmap(str(logo_path))
        if pm.isNull():
            return None
        # Scale first (fewer pixels to process)
        pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img = pm.toImage().convertToFormat(QImage.Format_ARGB32)
        for y in range(img.height()):
            for x in range(img.width()):
                c = QColor(img.pixel(x, y))
                r, g, b = c.red(), c.green(), c.blue()
                brightness = (r + g + b) / 3
                if brightness > 180:
                    # White background → transparent
                    img.setPixel(x, y, QColor(0, 0, 0, 0).rgba())
                else:
                    # Dark foreground → white for dark theme
                    alpha = max(0, min(255, int((180 - brightness) / 180 * 255)))
                    img.setPixel(x, y, QColor(255, 255, 255, alpha).rgba())
        return QPixmap.fromImage(img)

    # ── UI Construction ───────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(10)

        root.addWidget(self._make_header())
        root.addWidget(self._make_kpi_row())
        root.addWidget(self._make_main_content(), stretch=1)
        root.addWidget(self._make_data_section())

        self._make_status_bar()

    # ── Header ────────────────────────────────────────────────
    def _make_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("header_frame")
        frame.setStyleSheet(
            f"QFrame#header_frame {{"
            f"  background: {Config.BG_PANEL}; border-radius: 10px;"
            f"  border-bottom: 2px solid {Config.ACCENT_COLOR};"
            f"}}"
        )
        frame.setMaximumHeight(64)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 6, 16, 6)

        # Logo image (white-on-transparent for dark theme)
        logo_lbl = QLabel()
        logo_pixmap = self._load_logo(40)
        if logo_pixmap:
            logo_lbl.setPixmap(logo_pixmap)
        else:
            logo_lbl.setText("📊")
            logo_lbl.setFont(QFont("Segoe UI Emoji", 20))
        logo_lbl.setFixedSize(44, 44)
        logo_lbl.setAlignment(Qt.AlignCenter)

        title = QLabel(Config.APP_NAME)
        title.setStyleSheet(
            f"color: {Config.TEXT_PRIMARY}; font-size: 18px; font-weight: 700;"
        )
        subtitle = QLabel(f"  v{Config.APP_VERSION}")
        subtitle.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 12px;")

        layout.addWidget(logo_lbl)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        # Author + model badge
        author_lbl = QLabel(f"👤 {Config.AUTHOR}")
        author_lbl.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 11px;")
        model_badge = QLabel("⚡ Qwen 3 · Groq")
        model_badge.setStyleSheet(
            f"color: {Config.ACCENT_COLOR}; font-size: 11px; font-weight: 600;"
            f"background: rgba(0,212,255,0.1); border-radius: 4px; padding: 2px 8px;"
        )
        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 11px;")

        layout.addWidget(author_lbl)
        layout.addSpacing(16)
        layout.addWidget(model_badge)
        layout.addSpacing(16)
        layout.addWidget(self._clock_lbl)

        return frame

    # ── KPI Cards ─────────────────────────────────────────────
    def _make_kpi_row(self) -> QScrollArea:
        kpis = self._pipeline.kpis
        cards_row = KPICardsRow(kpis)

        scroll = QScrollArea()
        scroll.setWidget(cards_row)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(140)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    # ── Main content (charts + AI panel) ─────────────────────
    def _make_main_content(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # Left: Tab widget with chart panels
        tabs = QTabWidget()
        tabs.setMinimumWidth(750)

        self._sales_tab   = SalesChartsPanel(self._pipeline)
        self._finance_tab = FinanceChartsPanel(self._pipeline)
        self._hr_tab      = HRChartsPanel(self._pipeline)
        self._ops_tab     = OpsChartsPanel(self._pipeline)

        tabs.addTab(self._wrap_scroll(self._sales_tab),   "📈  Sales")
        tabs.addTab(self._wrap_scroll(self._finance_tab), "💰  Finance")
        tabs.addTab(self._wrap_scroll(self._hr_tab),      "👥  HR")
        tabs.addTab(self._wrap_scroll(self._ops_tab),     "⚙️  Operations")

        tabs.currentChanged.connect(self._on_tab_changed)

        # Right: AI panel
        self._ai_panel = AIAssistantPanel(self._ai_client, self._pipeline)

        splitter.addWidget(tabs)
        splitter.addWidget(self._ai_panel)
        splitter.setSizes([820, 380])
        return splitter

    @staticmethod
    def _wrap_scroll(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    # ── Data table section ────────────────────────────────────
    def _make_data_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab selector for data table
        self._data_tabs = QTabWidget()
        self._data_tabs.setMaximumHeight(260)

        df_sales   = self._pipeline.sales_monthly
        df_finance = self._pipeline.finance[
            ["date", "revenue", "gross_profit", "ebitda", "net_profit",
             "budget_revenue", "operating_cash_flow"]
        ]
        df_hr  = self._pipeline.hr_monthly
        df_ops = self._pipeline.operations

        self._data_tabs.addTab(DataTableWidget(df_sales,   "Monthly Sales"),    "Sales Data")
        self._data_tabs.addTab(DataTableWidget(df_finance, "Finance P&L"),      "Finance Data")
        self._data_tabs.addTab(DataTableWidget(df_hr,      "HR Overview"),      "HR Data")
        self._data_tabs.addTab(DataTableWidget(df_ops,     "Operations KPIs"),  "Ops Data")

        layout.addWidget(self._data_tabs)
        return container

    # ── Status bar ────────────────────────────────────────────
    def _make_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        bar.showMessage(
            f"  {Config.APP_NAME}  |  Data: 24 months (Jan 2024 – Dec 2025)  "
            f"|  Model: {Config.QWEN_MODEL}  |  Author: {Config.AUTHOR}"
        )

    # ── Live clock ────────────────────────────────────────────
    def _start_clock(self) -> None:
        self._tick()
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(1000)

    def _tick(self) -> None:
        self._clock_lbl.setText(
            datetime.now().strftime("[%H:%M:%S]  %Y-%m-%d")
        )

    # ── Tab change handler ────────────────────────────────────
    def _on_tab_changed(self, idx: int) -> None:
        self._data_tabs.setCurrentIndex(idx)

    # ── Styles ────────────────────────────────────────────────
    def _apply_styles(self) -> None:
        full_style = MAIN_STYLE + KPI_CARD_STYLE + AI_PANEL_STYLE
        self.setStyleSheet(full_style)
