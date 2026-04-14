"""
AI for Internal Reporting & BI Platform
========================================
Entry point — run this script to launch the dashboard.

Author : Tuan Tran
Version: 1.0.0
"""

import sys
import os

# Ensure project root is on the path so `src` imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.main import launch


if __name__ == "__main__":
    launch()
