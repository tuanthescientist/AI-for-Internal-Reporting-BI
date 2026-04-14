"""
Dark-theme QSS stylesheet for the BI platform.
Author: Tuan Tran
"""

from src.config.settings import Config

C = Config  # shorthand

MAIN_STYLE = f"""
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {C.BG_DARK};
    color: {C.TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Inter', 'Arial', sans-serif;
    font-size: 13px;
}}

QLabel {{
    color: {C.TEXT_PRIMARY};
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {{
    background-color: {C.ACCENT_COLOR};
    color: #000000;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #33DCFF;
}}
QPushButton:pressed {{
    background-color: #0095B6;
}}
QPushButton:disabled {{
    background-color: #333;
    color: #666;
}}

QPushButton#danger_btn {{
    background-color: {C.DANGER_COLOR};
    color: #fff;
}}
QPushButton#secondary_btn {{
    background-color: {C.BG_PANEL};
    color: {C.TEXT_PRIMARY};
    border: 1px solid #30363D;
}}
QPushButton#secondary_btn:hover {{
    border: 1px solid {C.ACCENT_COLOR};
}}

/* ── Input fields ────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {C.BG_CARD};
    color: {C.TEXT_PRIMARY};
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 8px;
    selection-background-color: {C.ACCENT_COLOR};
    selection-color: #000;
}}
QLineEdit:focus, QTextEdit:focus {{
    border: 1px solid {C.ACCENT_COLOR};
}}

/* ── Tab Widget ──────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid #30363D;
    border-radius: 6px;
    background-color: {C.BG_CARD};
}}
QTabBar::tab {{
    background-color: {C.BG_PANEL};
    color: {C.TEXT_SECONDARY};
    padding: 10px 22px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background-color: {C.ACCENT_COLOR};
    color: #000000;
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    background-color: #21262D;
    color: {C.TEXT_PRIMARY};
}}

/* ── Scroll bars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    width: 8px;
    background: {C.BG_DARK};
}}
QScrollBar::handle:vertical {{
    background: #30363D;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C.ACCENT_COLOR};
}}
QScrollBar:horizontal {{
    height: 8px;
    background: {C.BG_DARK};
}}
QScrollBar::handle:horizontal {{
    background: #30363D;
    border-radius: 4px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}

/* ── Table ───────────────────────────────────────────────── */
QTableWidget {{
    background-color: {C.BG_CARD};
    alternate-background-color: {C.BG_PANEL};
    gridline-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 6px;
    selection-background-color: rgba(0, 212, 255, 0.2);
}}
QHeaderView::section {{
    background-color: {C.BG_PANEL};
    color: {C.TEXT_SECONDARY};
    padding: 8px;
    border: none;
    border-right: 1px solid #30363D;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QHeaderView::section:hover {{
    background-color: #21262D;
    color: {C.TEXT_PRIMARY};
}}
QTableWidget::item {{
    padding: 6px;
}}
QTableWidget::item:selected {{
    background-color: rgba(0, 212, 255, 0.2);
    color: {C.TEXT_PRIMARY};
}}

/* ── Splitter ────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: #30363D;
    width: 2px;
}}

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {{
    background-color: {C.BG_PANEL};
    color: {C.TEXT_SECONDARY};
    border-top: 1px solid #30363D;
    font-size: 11px;
}}

/* ── Combo box ───────────────────────────────────────────── */
QComboBox {{
    background-color: {C.BG_CARD};
    color: {C.TEXT_PRIMARY};
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 6px 10px;
}}
QComboBox:hover {{
    border: 1px solid {C.ACCENT_COLOR};
}}
QComboBox QAbstractItemView {{
    background-color: {C.BG_CARD};
    selection-background-color: rgba(0, 212, 255, 0.3);
}}

/* ── ToolTip ─────────────────────────────────────────────── */
QToolTip {{
    background-color: {C.BG_PANEL};
    color: {C.TEXT_PRIMARY};
    border: 1px solid {C.ACCENT_COLOR};
    padding: 4px 8px;
    border-radius: 4px;
}}

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {{
    background-color: {C.BG_CARD};
    border: 1px solid #30363D;
    border-radius: 4px;
    text-align: center;
    color: {C.TEXT_PRIMARY};
    height: 8px;
}}
QProgressBar::chunk {{
    background-color: {C.ACCENT_COLOR};
    border-radius: 4px;
}}
"""

KPI_CARD_STYLE = f"""
QFrame#kpi_card {{
    background-color: {C.BG_CARD};
    border: 1px solid #30363D;
    border-radius: 10px;
}}
QFrame#kpi_card:hover {{
    border: 1px solid {C.ACCENT_COLOR};
}}
"""

AI_PANEL_STYLE = f"""
QFrame#ai_panel {{
    background-color: {C.BG_PANEL};
    border-left: 2px solid #30363D;
}}
"""
