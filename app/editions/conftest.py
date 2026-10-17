import sys
from pathlib import Path

# The shared engine (label, page_sort, io_utils) lives two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reading_engine"))
