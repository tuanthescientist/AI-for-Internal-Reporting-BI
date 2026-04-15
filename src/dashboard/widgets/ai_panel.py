"""
AIAssistantPanel — streaming chat panel powered by Qwen 3.
Uses QThread so AI calls never block the Qt event loop.
Author: Tuan Tran
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget, QComboBox,
)

from src.config.settings import Config
from src.ai.rag_retriever import RAGRetriever, Passage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class _AIWorker(QThread):
    """Runs a streaming AI generator in a background thread."""

    chunk_received  = pyqtSignal(str)   # emitted for each streamed token
    finished        = pyqtSignal()
    error_occurred  = pyqtSignal(str)

    def __init__(self, generator_fn) -> None:
        super().__init__()
        self._gen_fn = generator_fn     # callable that returns a generator

    def run(self) -> None:
        try:
            for chunk in self._gen_fn():
                self.chunk_received.emit(chunk)
            self.finished.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
class AIAssistantPanel(QFrame):
    """
    Right-side AI panel with:
    • Streaming chat history display
    • Message input + Send button
    • Preset action buttons (Generate Report, Detect Anomalies, Get Insights)
    • Action selector (dept / domain context)
    """

    def __init__(self, ai_client, pipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ai_panel")
        self.setMinimumWidth(360)
        self.setMaximumWidth(480)
        self._ai = ai_client
        self._pipeline = pipeline
        self._worker: _AIWorker | None = None
        self._rag = RAGRetriever(pipeline)
        self._pending_passages: List[Passage] = []
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title_row = QHBoxLayout()
        icon = QLabel("🤖")
        icon.setFont(QFont("Segoe UI Emoji", 18))
        title_lbl = QLabel("AI Assistant")
        title_lbl.setStyleSheet(
            f"color: {Config.ACCENT_COLOR}; font-size: 15px; font-weight: 700;"
        )
        model_lbl = QLabel("Qwen 3 · 32B")
        model_lbl.setStyleSheet(
            f"color: {Config.TEXT_SECONDARY}; font-size: 10px;"
        )
        self._lang_lbl = QLabel("EN")
        self._lang_lbl.setToolTip(
            "Language auto-detected from your message.\n"
            "Send a message in Vietnamese or English to switch."
        )
        self._lang_lbl.setStyleSheet(
            "background: #21262D; color: #58A6FF; font-size: 10px;"
            "font-weight: 700; padding: 2px 6px; border-radius: 4px;"
        )
        title_row.addWidget(icon)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(model_lbl)
        title_row.addWidget(self._lang_lbl)
        layout.addLayout(title_row)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #30363D;")
        layout.addWidget(div)

        # Context selector
        ctx_row = QHBoxLayout()
        ctx_lbl = QLabel("Context:")
        ctx_lbl.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 11px;")
        self._context_combo = QComboBox()
        self._context_combo.addItems(["All Domains", "Sales", "Finance", "HR", "Operations"])
        ctx_row.addWidget(ctx_lbl)
        ctx_row.addWidget(self._context_combo)
        layout.addLayout(ctx_row)

        # Chat history
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setMinimumHeight(300)
        self._chat_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._chat_display.setStyleSheet(
            f"background: {Config.BG_DARK}; border: 1px solid #30363D; border-radius: 8px;"
            f"padding: 8px; font-size: 12px; line-height: 1.5;"
        )
        self._append_system_message(
            "Welcome! I'm your AI Business Intelligence assistant powered by **Qwen 3 32B**.\n\n"
            "Ask me anything about the data, or use the buttons below to generate reports, "
            "detect anomalies, or get executive insights."
        )
        layout.addWidget(self._chat_display)

        # Status indicator
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            f"color: {Config.SUCCESS_COLOR}; font-size: 10px;"
        )
        layout.addWidget(self._status_lbl)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a business question…")
        self._input.returnPressed.connect(self._send_message)
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(60)
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._input)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        # Action buttons
        btn_grid = QHBoxLayout()
        self._report_btn  = self._make_action_btn("📝 Report",   self._generate_report)
        self._anomaly_btn = self._make_action_btn("🚨 Anomalies", self._detect_anomalies)
        self._insight_btn = self._make_action_btn("💡 Insights",  self._get_insights)
        self._clear_btn   = QPushButton("🗑")
        self._clear_btn.setFixedWidth(36)
        self._clear_btn.setObjectName("secondary_btn")
        self._clear_btn.setToolTip("Clear conversation")
        self._clear_btn.clicked.connect(self._clear_history)
        btn_grid.addWidget(self._report_btn)
        btn_grid.addWidget(self._anomaly_btn)
        btn_grid.addWidget(self._insight_btn)
        btn_grid.addWidget(self._clear_btn)
        layout.addLayout(btn_grid)

    @staticmethod
    def _make_action_btn(text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("secondary_btn")
        btn.clicked.connect(slot)
        return btn

    # ── Messaging helpers ─────────────────────────────────────
    def _append_system_message(self, text: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(Config.ACCENT_COLOR))
        cursor.insertText(f"🤖 AI\n", fmt)
        fmt.setForeground(QColor(Config.TEXT_PRIMARY))
        cursor.insertText(text + "\n\n", fmt)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _append_user_message(self, text: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(Config.WARNING_COLOR))
        cursor.insertText(f"👤 You\n", fmt)
        fmt.setForeground(QColor(Config.TEXT_SECONDARY))
        cursor.insertText(text + "\n\n", fmt)
        fmt.setForeground(QColor(Config.ACCENT_COLOR))
        cursor.insertText("🤖 AI\n", fmt)
        fmt.setForeground(QColor(Config.TEXT_PRIMARY))
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()
        self._response_start_format = fmt

    def _append_ai_chunk(self, chunk: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(Config.TEXT_PRIMARY))
        cursor.insertText(chunk, fmt)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _finalize_response(self) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(Config.TEXT_PRIMARY))
        cursor.insertText("\n", fmt)

        # Render RAG citation block if passages were used
        if self._pending_passages:
            self._append_citations(self._pending_passages)
            self._pending_passages = []

        cursor.movePosition(QTextCursor.End)
        fmt.setForeground(QColor(Config.TEXT_PRIMARY))
        cursor.insertText("\n", fmt)
        self._chat_display.setTextCursor(cursor)
        self._update_lang_label()
        self._set_busy(False)

    def _append_citations(self, passages: List[Passage]) -> None:
        """Render a styled 'Sources' block after the AI response."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()

        # Header
        fmt.setForeground(QColor("#6E7681"))
        cursor.insertText("─" * 38 + "\n", fmt)
        fmt.setForeground(QColor("#58A6FF"))
        cursor.insertText("📎 Sources\n", fmt)

        # One line per passage
        _DOMAIN_COLORS = {
            "Finance":    "#3FB950",
            "Sales":      "#00D4FF",
            "HR":         "#D29922",
            "Operations": "#A371F7",
            "Customers":  "#F0883E",
        }
        for p in passages:
            color = _DOMAIN_COLORS.get(p.domain, "#8B949E")
            fmt.setForeground(QColor(color))
            cursor.insertText(f"  {p.ref} ", fmt)
            fmt.setForeground(QColor("#8B949E"))
            cursor.insertText(f"{p.domain} — {p.source}\n", fmt)

        fmt.setForeground(QColor("#6E7681"))
        cursor.insertText("─" * 38 + "\n", fmt)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _update_lang_label(self) -> None:
        lang = self._ai.current_language
        if lang == "vi":
            self._lang_lbl.setText("VI")
            self._lang_lbl.setStyleSheet(
                "background: #1B3A2F; color: #3FB950; font-size: 10px;"
                "font-weight: 700; padding: 2px 6px; border-radius: 4px;"
            )
        else:
            self._lang_lbl.setText("EN")
            self._lang_lbl.setStyleSheet(
                "background: #21262D; color: #58A6FF; font-size: 10px;"
                "font-weight: 700; padding: 2px 6px; border-radius: 4px;"
            )

    # ── AI Actions ────────────────────────────────────────────
    def _send_message(self) -> None:
        msg = self._input.text().strip()
        if not msg or self._worker and self._worker.isRunning():
            return
        self._input.clear()

        # RAG retrieval — find relevant data passages for this query
        passages = self._rag.retrieve(msg, k=4)
        self._pending_passages = passages

        self._append_user_message(msg)
        self._set_busy(True)

        if passages:
            def gen():
                return self._ai.chat_rag(msg, passages)
        else:
            # Fallback: no relevant passages → plain chat with domain context
            data_ctx = self._build_data_context(self._context_combo.currentText())
            def gen():
                return self._ai.chat(msg, data_context=data_ctx)

        self._run_worker(gen)

    def _generate_report(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._append_user_message("📝 Generate comprehensive executive report")
        self._set_busy(True)
        summary = self._pipeline.ai_summary()

        def gen():
            return self._ai.generate_executive_report(summary)

        self._run_worker(gen)

    def _detect_anomalies(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        from src.ai.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(self._ai)

        domain = self._context_combo.currentText()
        p = self._pipeline

        if domain == "Finance":
            df = p.finance
            fin_candidates = ["revenue", "net_profit", "ebitda",
                               "total_opex", "operating_expenses"]
            cols = [c for c in fin_candidates if c in df.columns]
        elif domain == "HR":
            df = p.hr_monthly
            hr_candidates = ["headcount", "attrition_rate", "new_hires", "eNPS"]
            cols = [c for c in hr_candidates if c in df.columns]
        elif domain == "Operations":
            df = p.operations
            ops_candidates = ["process_efficiency_pct", "sla_compliance_pct",
                               "defect_rate_pct", "server_uptime_pct",
                               "change_failure_rate_pct"]
            cols = [c for c in ops_candidates if c in df.columns]
        else:
            df = p.operations
            ops_candidates = ["process_efficiency_pct", "sla_compliance_pct",
                               "defect_rate_pct", "server_uptime_pct"]
            cols = [c for c in ops_candidates if c in df.columns]

        anomalies = detector.detect_dataframe(df, cols)
        self._append_user_message(f"🚨 Detect anomalies — {domain}")
        self._set_busy(True)

        def gen():
            return self._ai.explain_anomalies(anomalies)

        self._run_worker(gen)

    def _get_insights(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._append_user_message("💡 Generate KPI insights")
        self._set_busy(True)
        kpis = self._pipeline.kpis

        def gen():
            return self._ai.get_kpi_insights(kpis)

        self._run_worker(gen)

    def _clear_history(self) -> None:
        self._chat_display.clear()
        self._ai.clear_history()
        self._append_system_message("Conversation cleared. How can I help you?")

    # ── Worker management ─────────────────────────────────────
    def _run_worker(self, gen_fn) -> None:
        self._worker = _AIWorker(gen_fn)
        self._worker.chunk_received.connect(self._append_ai_chunk)
        self._worker.finished.connect(self._finalize_response)
        self._worker.error_occurred.connect(self._handle_error)
        self._worker.start()

    def _handle_error(self, msg: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(Config.DANGER_COLOR))
        cursor.insertText(f"\n⚠️ Error: {msg}\n\n", fmt)
        self._chat_display.setTextCursor(cursor)
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._send_btn.setEnabled(not busy)
        self._report_btn.setEnabled(not busy)
        self._anomaly_btn.setEnabled(not busy)
        self._insight_btn.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._status_lbl.setText(
            "⏳ Qwen 3 thinking…" if busy else "✅ Ready"
        )
        self._status_lbl.setStyleSheet(
            f"color: {Config.WARNING_COLOR}; font-size: 10px;"
            if busy
            else f"color: {Config.SUCCESS_COLOR}; font-size: 10px;"
        )

    def _build_data_context(self, domain: str) -> str:
        """Build minimal, token-efficient data context for AI prompts."""
        import json

        p = self._pipeline

        def _safe_cols(df, *candidates):
            return [c for c in candidates if c in df.columns]

        if domain == "Sales":
            sal = p.sales_monthly
            rev_col = "revenue"  # already normalised in sales_monthly
            cols = _safe_cols(sal, "date", rev_col, "units_sold", "gross_margin_pct")
            data = {
                "sales_last_3": sal[cols].tail(3).to_dict("records"),
                "top_regions":  p.sales_by_region.head(3).to_dict("records"),
                "top_products": p.sales_by_product.head(3)[["product", "revenue"]].to_dict("records"),
            }
        elif domain == "Finance":
            fin = p.finance
            cols = _safe_cols(fin, "date", "revenue", "net_profit", "ebitda",
                              "gross_margin_pct", "ebitda_margin_pct")
            data = {"finance_last_3": fin[cols].tail(3).to_dict("records")}
        elif domain == "HR":
            hr = p.hr_monthly
            cols = _safe_cols(hr, "date", "headcount", "attrition_rate", "new_hires", "eNPS")
            data = {"hr_last_3": hr[cols].tail(3).to_dict("records")}
        elif domain == "Operations":
            ops = p.operations
            cols = _safe_cols(ops, "date", "process_efficiency_pct", "sla_compliance_pct",
                              "defect_rate_pct", "server_uptime_pct",
                              "deploy_frequency_month", "customer_satisfaction")
            data = {"ops_last_3": ops[cols].tail(3).to_dict("records")}
        else:
            data = {"kpis": p.kpis}

        try:
            return json.dumps(data, separators=(',', ':'), default=str)
        except Exception:
            return str(data)
