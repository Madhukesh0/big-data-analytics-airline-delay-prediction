"""Run the Streamlit app through AppTest and assert no exceptions in any view.

This is a smoke test: it does not click train/sim buttons (those would start
Spark again); it verifies that every view of the already-populated project
renders without raising a Python exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _run_all_views():
    for view in ["Overview", "Data & ETL", "Train & Evaluate", "Predictions", "Live Simulation", "History"]:
        at = AppTest.from_file(str(APP), default_timeout=240)
        at.run()
        assert not at.exception, f"[{view}] exceptions: {list(at.exception)}"
        at.sidebar.radio[0].set_value(view)
        at.run()
        assert not at.exception, f"[{view}] view exceptions: {list(at.exception)}"
        print(f"OK: {view}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--overview-only":
        at = AppTest.from_file(str(APP), default_timeout=240)
        at.run()
        assert not at.exception, f"exceptions: {list(at.exception)}"
        print("OK: Overview only")
    else:
        _run_all_views()
    print("STREAMLIT SMOKE TEST PASSED")
