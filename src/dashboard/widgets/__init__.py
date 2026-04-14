# src/dashboard/widgets/__init__.py
from .kpi_cards import KPICard, KPICardsRow
from .charts import ChartCanvas, SalesChartsPanel, FinanceChartsPanel, HRChartsPanel, OpsChartsPanel
from .ai_panel import AIAssistantPanel
from .data_table import DataTableWidget

__all__ = [
    "KPICard", "KPICardsRow",
    "ChartCanvas",
    "SalesChartsPanel", "FinanceChartsPanel", "HRChartsPanel", "OpsChartsPanel",
    "AIAssistantPanel",
    "DataTableWidget",
]
