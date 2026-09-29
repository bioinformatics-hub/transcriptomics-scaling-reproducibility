from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    if str(PLOTTING_DIR) in sys.path:
        sys.path.remove(str(PLOTTING_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))

import polars as pl

from plotting.scgpt_batch import (
    _select_most_complete_runs,
    create_plot,
    export_chunk_snapshot,
)
from plotting.training_loss import GENEFORMER_BCE_CONFIG
from plotting.utils import PLOTS_DIR, ROOT_DIR


DEFAULT_AIM_REPO_ROOT = ROOT_DIR / "aim-repo" / "geneformer_batch"
DEFAULT_AIM_REPO = (
    DEFAULT_AIM_REPO_ROOT / "geneformer_batch-repo"
    if (DEFAULT_AIM_REPO_ROOT / "geneformer_batch-repo").exists()
    else DEFAULT_AIM_REPO_ROOT
)
DEFAULT_OUTPUT_DIR = PLOTS_DIR / "geneformer_batch"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "01_training_loss_metrics.csv"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "01_training_loss.svg"

# The observed sweep spans approximately 5.3--11.2 cross-entropy.
# These fixed limits preserve the generous padding used by the equivalent scGPT plot.
Y_LIMITS = (4.2, 10.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the geneformer_batch training cross-entropy sweep in the "
            "same style as scgpt_batch."
        )
    )
    parser.add_argument("--aim-repo", type=Path, default=DEFAULT_AIM_REPO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the current Aim chunk snapshot even when the CSV exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv.exists() and not args.force:
        df = pl.read_csv(args.csv, infer_schema_length=10_000)
    else:
        df = export_chunk_snapshot(
            args.aim_repo,
            args.csv,
            metric_name=GENEFORMER_BCE_CONFIG.metric_name,
            model_name="Geneformer",
        )

    plot_df = _select_most_complete_runs(df)
    create_plot(
        plot_df,
        args.output,
        title=GENEFORMER_BCE_CONFIG.title,
        ylabel=GENEFORMER_BCE_CONFIG.ylabel,
        y_limits=Y_LIMITS,
        line_width=1.0,
    )
    run_count = df.select("run_hash").n_unique()
    plotted_run_count = plot_df.select("run_hash").n_unique()
    max_step = int(plot_df["step"].max())
    print(f"Saved {args.output}")
    print(
        f"Exported {run_count} traces and plotted {plotted_run_count} configurations; "
        f"maximum observed step: {max_step}"
    )


if __name__ == "__main__":
    main()
