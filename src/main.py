"""
Application entry point.
Author: Tuan Tran
"""

from __future__ import annotations

import logging
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from src.config.settings import Config


def launch() -> None:
    """Initialise and run the PyQt5 BI dashboard."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    Config.ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName("Tuan Tran")

    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    # Import here to keep startup fast
    from src.dashboard.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
