"""Locations and reproducible packaging of the paper's processed input tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "publication" / "plot_data"
INPUTS = (
    "geneformer_batch/01_training_loss_metrics.csv",
    "scgpt_batch/01_training_loss_metrics.csv",
    "geneformer_like/01_training_loss_prepared.parquet",
    "geneformer_like/08_downstream_metrics_prepared.parquet",
    "scgpt_like/01_training_loss_prepared.parquet",
    "scgpt_like/08_downstream_metrics_prepared.parquet",
    "geneformer_lr/01_lr_sampled_prepared.parquet",
    "scgpt_lr_mse/01_lr_sampled_prepared.parquet",
    "geneformer_dw/01_isoflops_prepared.parquet",
    "geneformer_dw/02_training_loss_flops_prepared.parquet",
    "scgpt_dw_rerun/01_isoflops_prepared.parquet",
    "scgpt_dw_rerun/02_training_loss_flops_prepared.parquet",
)


def paper_data(relative_path: str) -> Path:
    """Use the versioned snapshot independently of local exploratory caches."""
    return DATA_DIR / Path(relative_path).with_suffix(".parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_data(source_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for relative in INPUTS:
        source = source_dir / relative
        frame = (
            pl.read_csv(source, infer_schema_length=10_000)
            if source.suffix == ".csv"
            else pl.read_parquet(source)
        )
        # Host details are not analysis inputs. Preserve all numerical observations.
        removed = [
            c
            for c in frame.columns
            if c.startswith("paths.")
            or c in {"metadata.user", "metadata.machine", "source_csv"}
        ]
        frame = frame.drop(removed)
        target = output_dir / Path(relative).with_suffix(".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target, compression="zstd", compression_level=12)
        from polars.testing import assert_frame_equal

        assert_frame_equal(frame, pl.read_parquet(target))
        manifest.append(
            {
                "source": relative,
                "source_sha256": sha256(source),
                "file": str(target.relative_to(output_dir)),
                "sha256": sha256(target),
                "rows": frame.height,
                "columns": frame.columns,
                "removed_columns": removed,
            }
        )
        print(
            f"{target.name}: {frame.height:,} rows, {target.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "plots")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for the snapshot."
    )
    args = parser.parse_args()
    package_data(args.source, args.output)


if __name__ == "__main__":
    main()
