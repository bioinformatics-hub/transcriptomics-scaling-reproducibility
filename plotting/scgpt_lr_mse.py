from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    if str(PLOTTING_DIR) in sys.path:
        sys.path.remove(str(PLOTTING_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting import geneformer_lr as lr
from plotting.training_loss import SCGPT_MSE_CONFIG
from plotting.utils import PLOTS_DIR, ROOT_DIR


RUN_NAME = "scgpt_lr"
OUTPUT_NAME = "scgpt_lr_mse"


def _configure_scgpt_defaults() -> None:
    output_dir = PLOTS_DIR / OUTPUT_NAME

    lr.DEFAULT_AIM_REPO = ROOT_DIR / "aim-repo" / RUN_NAME
    lr.DEFAULT_CSV = output_dir / "01_lr_sampled_metrics.csv"
    lr.DEFAULT_PREPARED_DATA = output_dir / "01_lr_sampled_prepared.parquet"
    lr.DEFAULT_SWEEP_OUTPUT = output_dir / "01_lr_sweeps.svg"
    lr.DEFAULT_PARAM_SCALING_OUTPUT = output_dir / "02_optimal_lr_vs_parameters.svg"
    lr.DEFAULT_DEPTH_SCALING_OUTPUT = output_dir / "03_optimal_lr_vs_depth.svg"
    lr.DEFAULT_WIDTH_SCALING_OUTPUT = output_dir / "04_optimal_lr_vs_width.svg"
    lr.DEFAULT_POWER_LAW_COMPARISON_OUTPUT = output_dir / "05_lr_power_law_model_comparison.svg"
    lr.DEFAULT_POWER_LAW_BOOTSTRAP_OUTPUT = output_dir / "06_lr_power_law_bootstrap.svg"
    lr.DEFAULT_POWER_LAW_SUMMARY_OUTPUT = output_dir / "07_lr_power_law_summary.svg"
    lr.DEFAULT_FORMULA_LR_TABLE_OUTPUT = output_dir / "08_formula_lr_table.svg"
    lr.DEFAULT_FORMULA_LR_TABLE_CSV_OUTPUT = output_dir / "08_formula_lr_table.csv"
    lr.DEFAULT_POWER_LAWS_COMPUTE_SLICES_OUTPUT = output_dir / "09_power_laws_comparison_across_compute_slices.svg"
    lr.DEFAULT_TARGET_TRAINING_FLOPS = 1.4e17

    lr.RUN_QUERY = "run.config.metadata.title == 'scgpt_lr'"
    lr.METRICS_TO_EXTRACT = {SCGPT_MSE_CONFIG.metric_name}
    lr.GENEFORMER_BCE_CONFIG = SCGPT_MSE_CONFIG


def main() -> None:
    _configure_scgpt_defaults()
    lr.main()


if __name__ == "__main__":
    main()
