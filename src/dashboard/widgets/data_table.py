"""
DataTableWidget — sortable, filterable table for any Pandas DataFrame.
Author: Tuan Tran
"""

from __future__ import annotations

import pandas as pd
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableView, QVBoxLayout, QWidget,
)
from src.config.settings import Config


class _PandasModel(QAbstractTableModel):
    """Qt model backed by a Pandas DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iloc[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(val, float):
                return f"{val:,.2f}"
            if isinstance(val, int):
                return f"{val:,}"
            return str(val)
        if role == Qt.TextAlignmentRole:
            if isinstance(val, (int, float)):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        return None

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section]).replace("_", " ").title()
            return str(section + 1)
        return None


class DataTableWidget(QWidget):
    """
    Widget encapsulating a QTableView with:
    • Live filter search box
    • Row / column count label
    • Export to CSV button
    """

    def __init__(self, df: pd.DataFrame, title: str = "Data", parent=None) -> None:
        super().__init__(parent)
        self._df = df
        self._title = title
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        # ── Toolbar ───────────────────────────────────────────
        toolbar = QHBoxLayout()
        title_lbl = QLabel(self._title)
        title_lbl.setStyleSheet(
            f"color: {Config.TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Filter…")
        self._search.setMaximumWidth(200)
        self._search.textChanged.connect(self._on_filter)
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet(f"color: {Config.TEXT_SECONDARY}; font-size: 11px;")
        export_btn = QPushButton("⬇ Export CSV")
        export_btn.setObjectName("secondary_btn")
        export_btn.setFixedWidth(110)
        export_btn.clicked.connect(self._export)
        toolbar.addWidget(title_lbl)
        toolbar.addStretch()
        toolbar.addWidget(self._count_lbl)
        toolbar.addWidget(self._search)
        toolbar.addWidget(export_btn)
        layout.addLayout(toolbar)

        # ── Table ─────────────────────────────────────────────
        self._source_model = _PandasModel(self._df)
        self._proxy_model = QSortFilterProxyModel()
        self._proxy_model.setSourceModel(self._source_model)
        self._proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy_model.setFilterKeyColumn(-1)   # search all columns

        self._table = QTableView()
        self._table.setModel(self._proxy_model)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(28)
        self._table.setShowGrid(True)
        layout.addWidget(self._table)

        self._update_count()

    def _on_filter(self, text: str) -> None:
        self._proxy_model.setFilterFixedString(text)
        self._update_count()

    def _update_count(self) -> None:
        visible = self._proxy_model.rowCount()
        total = self._source_model.rowCount()
        self._count_lbl.setText(f"{visible:,} / {total:,} rows")

    def _export(self) -> None:
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self._title}.csv", "CSV Files (*.csv)"
        )
        if path:
            self._df.to_csv(path, index=False)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the displayed DataFrame."""
        self._df = df
        self._source_model = _PandasModel(df)
        self._proxy_model.setSourceModel(self._source_model)
        self._update_count()
