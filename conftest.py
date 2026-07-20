"""
Conftest la radacina proiectului.

Prezenta acestui fisier face ca pytest sa insereze radacina proiectului in
sys.path, astfel incat testele sa poata face `from src import ...` fara sa
fie nevoie sa instalam proiectul ca pachet (pip install -e .).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
