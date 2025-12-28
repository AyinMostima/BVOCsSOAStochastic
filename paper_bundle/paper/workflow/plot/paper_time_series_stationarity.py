from __future__ import annotations

import os
import subprocess
import sys

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "time_series_stationarity.py"
TARGETS = [
    FIGURE_DIR / "SOA_timeseries_stationarity.png",
    FIGURE_DIR / "SOA_delta_timeseries_combined.png",
]


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=BUNDLE_ROOT, check=True, env=env)
    missing = [p for p in TARGETS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected stationarity figures: {missing}")
    print("Time series stationarity figures written directly to paper\\figure.")


if __name__ == "__main__":
    sys.exit(main())
