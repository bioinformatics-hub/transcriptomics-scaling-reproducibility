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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from aim.storage.rockscontainer import RocksContainer

from plotting.utils import (
    COLOR_SHADES,
    PLOTS_DIR,
    ROOT_DIR,
    apply_plot_style,
    ensure_parent_dir,
    exponential_moving_average,
    remove_bounding_box,
    save_figure,
    series_unique_sorted,
)


DEFAULT_AIM_REPO_ROOT = ROOT_DIR / "aim-repo" / "scgpt_batch"
DEFAULT_AIM_REPO = (
    DEFAULT_AIM_REPO_ROOT / "scgpt_batch-repo"
    if (DEFAULT_AIM_REPO_ROOT / "scgpt_batch-repo").exists()
    else DEFAULT_AIM_REPO_ROOT
)
DEFAULT_OUTPUT_DIR = PLOTS_DIR / "scgpt_batch"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "01_training_loss_metrics.csv"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "01_training_loss.svg"

PARAM_COLOR = "datamodule.batch_size"
PARAM_SHADE = "model.d_model"
EMA_ALPHA = 0.05
X_MIN = -500
Y_MIN = 200 * 0.8
Y_MAX = 450 * 1.2

FONT_SIZE_TICKS = 24
FONT_SIZE_TEXT = 34
FONT_SIZE_LEGEND = 22

CONFIG_FIELDS = {
    "metadata.pipeline": ("metadata", "pipeline"),
    "metadata.run_name": ("metadata", "run_name"),
    "metadata.title": ("metadata", "title"),
    "datamodule.batch_size": ("datamodule", "batch_size"),
    "model.context_length": ("model", "context_length"),
    "model.d_model": ("model", "d_model"),
    "model.transformer.n_layers": ("model", "transformer", "n_layers"),
    "trainer.accumulate_grad": ("trainer", "accumulate_grad"),
    "trainer.n_steps": ("trainer", "n_steps"),
}

RUN_FIELDS = {
    "hparams.parameters_total": ("parameters_total",),
    "hparams.parameters_embedding": ("parameters_embedding",),
    "hparams.parameters_encoder": ("parameters_encoder",),
    "hparams.parameters_decoder": ("parameters_decoder",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the scgpt_batch training-MSE sweep in the project figure style."
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


def _chunk_tree(container: RocksContainer, kind: str, run_hash: str):
    path = (kind, "v2", "chunks", run_hash) if kind == "seqs" else (kind, "chunks", run_hash)
    return container.tree().subtree(path)


def _context_ids(meta_tree) -> list[int]:
    return sorted(
        {
            path[1]
            for path in meta_tree.keys((), level=2)
            if len(path) >= 2 and path[0] == "contexts"
        }
    )


def _extract_run(
    meta_path: Path, seq_path: Path, *, metric_name: str = "mse"
) -> list[pl.DataFrame]:
    run_hash = meta_path.name
    meta_container = RocksContainer(str(meta_path), read_only=True)
    seq_container = RocksContainer(str(seq_path), read_only=True)
    try:
        meta_tree = _chunk_tree(meta_container, "meta", run_hash)
        seq_tree = _chunk_tree(seq_container, "seqs", run_hash)
        metadata = {
            column: meta_tree.get(("attrs", "config", *path), None)
            for column, path in CONFIG_FIELDS.items()
        }
        metadata.update(
            {
                column: meta_tree.get(("attrs", "hparams", *path), None)
                for column, path in RUN_FIELDS.items()
            }
        )

        frames: list[pl.DataFrame] = []
        for context_id in _context_ids(meta_tree):
            if meta_tree.get(("contexts", context_id, "subset"), None) != "train":
                continue
            if (
                meta_tree.get(
                    ("traces", context_id, metric_name, "last_step"), None
                )
                is None
            ):
                continue

            arrays = {
                name: seq_tree.array((context_id, metric_name, name)).values_numpy()
                for name in ("step", "epoch", "time", "val")
            }
            length = min(len(values) for values in arrays.values())
            if length == 0:
                continue

            frame = pl.DataFrame(
                {
                    "run_hash": [run_hash] * length,
                    "metric_name": [metric_name] * length,
                    "context.subset": ["train"] * length,
                    "step": arrays["step"][:length],
                    "epoch": arrays["epoch"][:length],
                    "timestamp": arrays["time"][:length],
                    "value": arrays["val"][:length],
                    **{column: [value] * length for column, value in metadata.items()},
                }
            )
            frames.append(frame)
        return frames
    finally:
        meta_container.close()
        seq_container.close()


def export_chunk_snapshot(
    aim_repo: Path,
    csv_path: Path,
    *,
    metric_name: str = "mse",
    model_name: str = "scGPT",
) -> pl.DataFrame:
    aim_root = aim_repo / ".aim"
    meta_root = aim_root / "meta" / "chunks"
    seq_root = aim_root / "seqs" / "chunks"
    if not meta_root.exists() or not seq_root.exists():
        raise FileNotFoundError(f"Aim chunk directories not found under {aim_repo}")

    frames: list[pl.DataFrame] = []
    for meta_path in sorted(meta_root.iterdir()):
        seq_path = seq_root / meta_path.name
        if meta_path.is_dir() and seq_path.is_dir():
            frames.extend(
                _extract_run(meta_path, seq_path, metric_name=metric_name)
            )
    if not frames:
        raise ValueError(
            f"No {model_name} training-{metric_name} traces found under {aim_repo}"
        )

    df = pl.concat(frames, how="diagonal_relaxed").sort(["run_hash", "step"])
    ensure_parent_dir(csv_path)
    df.write_csv(csv_path)
    return df


def _legend_handles(
    df: pl.DataFrame,
) -> tuple[list[object], list[mlines.Line2D], list[mlines.Line2D]]:
    batch_sizes = series_unique_sorted(df, PARAM_COLOR)
    batch_handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps[COLOR_SHADES[index]](0.7),
            linewidth=3,
            label=f"Batch size: {int(batch_size)}",
        )
        for index, batch_size in enumerate(batch_sizes)
    ]

    embedding_dims = series_unique_sorted(df, PARAM_SHADE)
    grey_values = np.linspace(0.3, 1.0, len(embedding_dims))
    embedding_handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps["Greys"](grey_values[index]),
            linewidth=3,
            label=f"Emb. dim: {int(embedding_dim)}",
        )
        for index, embedding_dim in enumerate(embedding_dims)
    ]
    return batch_sizes, batch_handles, embedding_handles


def _select_most_complete_runs(df: pl.DataFrame) -> pl.DataFrame:
    selected_hashes = (
        df.group_by([PARAM_COLOR, PARAM_SHADE, "run_hash"])
        .agg(
            pl.col("step").max().alias("max_step"),
            pl.len().alias("point_count"),
        )
        .sort(
            [PARAM_COLOR, PARAM_SHADE, "max_step", "point_count", "run_hash"],
            descending=[False, False, True, True, False],
        )
        .group_by([PARAM_COLOR, PARAM_SHADE], maintain_order=True)
        .head(1)
        .select("run_hash")
        .to_series()
        .to_list()
    )
    return df.filter(pl.col("run_hash").is_in(selected_hashes))


def create_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    title: str = "Training Performance on Binned Gene Expression",
    ylabel: str = "MSE",
    y_limits: tuple[float, float] = (Y_MIN, Y_MAX),
    line_width: float = 0.6,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    batch_sizes, batch_handles, embedding_handles = _legend_handles(df)
    embedding_dims = series_unique_sorted(df, PARAM_SHADE)
    shade_by_embedding = dict(
        zip(embedding_dims, np.linspace(0.3, 1.0, len(embedding_dims)), strict=True)
    )

    for batch_index, batch_size in enumerate(batch_sizes):
        batch_df = df.filter(pl.col(PARAM_COLOR) == batch_size)
        cmap = plt.colormaps[COLOR_SHADES[batch_index]]
        for run_hash in series_unique_sorted(batch_df, "run_hash"):
            run_df = (
                batch_df.filter(pl.col("run_hash") == run_hash)
                .group_by("step")
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.col(PARAM_SHADE).first().alias(PARAM_SHADE),
                )
                .sort("step")
            )
            embedding_dim = run_df[PARAM_SHADE][0]
            ax.plot(
                run_df["step"].to_numpy(),
                exponential_moving_average(
                    run_df["value"].to_numpy(), alpha=EMA_ALPHA
                ),
                color=cmap(shade_by_embedding[embedding_dim]),
                linewidth=line_width,
                zorder=3,
            )

    ax.set_xlim(X_MIN, 15_000)
    ax.set_ylim(*y_limits)
    ax.margins(x=0.02, y=0.03)
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    ax.set_title(
        title,
        fontsize=FONT_SIZE_TEXT,
        y=1.03,
    )
    ax.set_xlabel("Training Steps", fontsize=FONT_SIZE_TEXT)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_TEXT)

    batch_legend = ax.legend(
        handles=batch_handles,
        loc="upper left",
        title="Batch Size",
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND,
    )
    ax.add_artist(batch_legend)
    ax.legend(
        handles=embedding_handles,
        loc="upper right",
        title="Embedding Dimension",
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND,
    )
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    if args.csv.exists() and not args.force:
        df = pl.read_csv(args.csv, infer_schema_length=10_000)
    else:
        df = export_chunk_snapshot(args.aim_repo, args.csv)

    plot_df = _select_most_complete_runs(df)
    create_plot(plot_df, args.output)
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
