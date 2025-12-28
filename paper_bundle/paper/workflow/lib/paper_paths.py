
from pathlib import Path
import sys

# Bundle-local roots for portable execution.
BUNDLE_ROOT = Path(__file__).resolve().parents[3]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

PAPER_ROOT = BUNDLE_ROOT / "paper"
WORKFLOW_DIR = PAPER_ROOT / "workflow"
CHECKPOINT_DIR = PAPER_ROOT / "checkpoint"
FIGURE_DIR = PAPER_ROOT / "figure"
FIGURES_DIR = BUNDLE_ROOT / "figures"
INTERMEDIATE_DIR = BUNDLE_ROOT / "intermediate"
TABLES_DIR = BUNDLE_ROOT / "tables"

for path in (WORKFLOW_DIR, CHECKPOINT_DIR, FIGURE_DIR, FIGURES_DIR, INTERMEDIATE_DIR, TABLES_DIR):
    path.mkdir(parents=True, exist_ok=True)


def _register_local_fonts() -> None:
    try:
        from paper.workflow.lib.plot_style_helvetica import register_helvetica_fonts
    except Exception:
        return
    try:
        register_helvetica_fonts(BUNDLE_ROOT.parent)
    except Exception:
        return


_register_local_fonts()
