"""Merge the completed ranked-grid rerun into local analysis caches.

Run after downloading geneformer_like_missing-repo to aim-repo/geneformer_like_missing
and its config/downstream CSV into checkpoints/geneformer_like/<architecture>/.
The original Aim repository is retained; rerun this module after any fresh export.
"""

from argparse import Namespace

import polars as pl

from plotting import geneformer_like as g


def main() -> None:
    supplemental_csv = g.DEFAULT_CSV.parent / "missing_metrics.csv"
    g.export_metrics_csv(
        g.ROOT_DIR / "aim-repo" / "geneformer_like_missing",
        supplemental_csv,
        metrics_to_extract=g.METRICS_TO_EXTRACT,
        run_query=g.RUN_QUERY,
        config_keys=g.CONFIG_KEYS,
        run_fields=g.RUN_FIELDS,
    )
    extra = pl.read_csv(supplemental_csv)
    architecture = g.FLOPS_JOIN_COLUMNS
    assert extra.select(architecture).unique().height == 1
    # A resumed logger has a new hash and can replay steps since its checkpoint.
    # Keep the later segment throughout the overlap, then assign one logical ID.
    latest = extra.sort("timestamp").tail(1)["run_hash"][0]
    keys = architecture + ["metric_name", "step"] + [
        column for column in extra.columns if column.startswith("context.")
    ]
    extra = extra.sort("timestamp").unique(subset=keys, keep="last").with_columns(
        pl.lit(latest).alias("run_hash")
    )
    original = pl.read_csv(g.DEFAULT_CSV)
    retained = original.join(extra.select(architecture).unique(), on=architecture, how="anti")
    combined = pl.concat([retained, extra], how="diagonal_relaxed")
    assert combined.select(architecture).unique().height == 100
    combined.write_csv(g.DEFAULT_CSV)
    args = Namespace(
        force=True, prepared_data=g.DEFAULT_PREPARED_DATA,
        downstream_data=g.DEFAULT_DOWNSTREAM_DATA, checkpoint_dir=g.DEFAULT_CHECKPOINT_DIR,
    )
    prepared, _ = g._ensure_prepared_data(args, g.DEFAULT_CSV)
    downstream, _ = g._ensure_downstream_data(args, prepared)
    assert prepared["run_hash"].n_unique() == 100
    assert downstream["model_folder"].n_unique() == 100
    rerun = prepared.filter(pl.col("run_hash") == latest)
    assert rerun.height == 50000 and rerun["step"].n_unique() == 50000
    assert rerun["step"].min() == 0 and rerun["step"].max() == 49999
    print("Updated training and downstream caches: 100 ranked architectures.")


if __name__ == "__main__":
    main()
