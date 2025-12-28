from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = ROOT / "paper" / "checkpoint"
MATPLOTLIBRC = ROOT / "paper" / "matplotlibrc"


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(MATPLOTLIBRC)
    subprocess.run(["python", "其他VOC拟合测试.py"], cwd=ROOT, check=True, env=env)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "results_sci_format.xlsx"
    if src.exists():
        shutil.copy2(src, CHECKPOINT_DIR / "results_sci_format.xlsx")
    print("Other VOC fit results copied to paper\\checkpoint.")


if __name__ == "__main__":
    sys.exit(main())
