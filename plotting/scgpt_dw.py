from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    if str(PLOTTING_DIR) in sys.path:
        sys.path.remove(str(PLOTTING_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting import geneformer_dw as dw  # noqa: E402
from plotting.training_loss import SCGPT_MSE_CONFIG  # noqa: E402
from plotting.utils import ROOT_DIR  # noqa: E402


RUN_NAME = "scgpt_dw_rerun"


def _configure_scgpt_defaults() -> None:
    for name in dir(dw):
        value = getattr(dw, name)
        if name.startswith("DEFAULT_") and isinstance(value, Path):
            setattr(dw, name, Path(str(value).replace("geneformer_dw", RUN_NAME)))

    # Keep rerun caches separate so an earlier-rule export cannot be reused.
    dw.DEFAULT_AIM_REPO = ROOT_DIR / "aim-repo" / "scgpt_dw_rerun-repo"
    dw.DEFAULT_DOWNSTREAM_DIR = ROOT_DIR / "checkpoints" / RUN_NAME

    # Analyze seven log-spaced compute slices from 3e15 through the terminal
    # 1e18-FLOP budget. Keep this override scoped to scGPT; geneformer_dw
    # retains its existing defaults.
    dw.DEFAULT_MIN_ISOFLOP_FRACTION = 3.0e15 / dw.DEFAULT_TARGET_TRAINING_FLOPS
    dw.DEFAULT_LOG_SPACED_MIN_TRAINING_FLOPS = 3.0e15

    dw.RUN_TITLE = RUN_NAME
    dw.RUN_QUERY = f"run.config.metadata.title == '{RUN_NAME}'"
    dw.METRICS_TO_EXTRACT = {SCGPT_MSE_CONFIG.metric_name}
    dw.GENEFORMER_BCE_CONFIG = SCGPT_MSE_CONFIG


def main() -> None:
    _configure_scgpt_defaults()
    dw.main()


if __name__ == "__main__":
    main()
