"""Verify the downloaded DW rerun and report the paper's numerical diagnostics.

Run after ``python -m plotting.scgpt_dw --force``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
from aim import Repo

from plotting import geneformer_dw as dw
from plotting.aim import _chunk_run_tree, _collect_chunk_attrs
from plotting.utils import ROOT_DIR
from tools.paper_plots.supplementary import (
    _sample_dw_training_curves,
    _surface_bootstrap_instability,
)


def main() -> None:
    source = ROOT_DIR / "aim-repo/scgpt_dw_rerun-repo"
    repo = Repo(str(source))
    runs = []
    try:
        for run_hash in sorted(repo.list_all_runs()):
            config, _ = _collect_chunk_attrs(_chunk_run_tree(repo, run_hash))
            sweep = config["sweep_metadata"]
            params = sweep["target_non_embedding_params"]
            depth = sweep["target_depth"]
            expected_lr = 10 ** (
                -0.8694153396464286
                - 0.4337094217502787 * math.log10(params)
                + 0.33430541593634633 * math.log10(depth)
            )
            assert config["metadata"]["title"] == "scgpt_dw_rerun", run_hash
            assert math.isclose(config["model"]["lr"], expected_lr, rel_tol=1e-12), (
                run_hash
            )
            assert config["trainer"]["lr_scheduler"] == "warmup_constant", run_hash
            assert config["trainer"]["lr_warmup_steps"] == 500, run_hash
            runs.append(
                dict(
                    run_hash=run_hash,
                    target_params=params,
                    depth=depth,
                    lr=config["model"]["lr"],
                    n_steps=config["trainer"]["n_steps"],
                )
            )
    finally:
        repo.close()
    expected_grid = {
        (n, d)
        for n in (10**6, 3 * 10**6, 10**7, 3 * 10**7, 10**8, 3 * 10**8, 10**9)
        for d in (2, 4, 8, 16, 24, 32, 64)
    }
    assert {(r["target_params"], r["depth"]) for r in runs} == expected_grid
    report = {
        "source": str(source.relative_to(ROOT_DIR)),
        "segments": runs,
        "formulations": {},
    }
    for label, directory in [("ranked", "geneformer_dw"), ("binned", "scgpt_dw_rerun")]:
        base = ROOT_DIR / "plots" / directory
        data = pl.read_parquet(base / "01_isoflops_prepared.parquet")
        training = pl.read_parquet(base / "02_training_loss_flops_prepared.parquet")
        assert training["run_hash"].n_unique() == 49
        if label == "binned":
            assert set("+".join(training["run_hash"].unique()).split("+")) == {
                r["run_hash"] for r in runs
            }
            endpoints = dict(
                training.group_by("run_hash").agg(pl.col("step").max()).iter_rows()
            )
            for logical_hash, last_step in endpoints.items():
                requested = max(
                    r["n_steps"]
                    for r in runs
                    if r["run_hash"] in logical_hash.split("+")
                )
                assert last_step >= requested - 1, (logical_hash, last_step, requested)
        fit = dw._surface_fit_df(data)
        model = dw._fit_surface_model_from_fit_df(fit)
        early = dw._surface_fit_df(_sample_dw_training_curves(training))
        combined = pl.concat([fit, early], how="diagonal_relaxed")
        summary = {}
        for name, frame, fitted in [
            ("analysis", fit, model),
            ("early", early, model),
            ("combined", combined, dw._fit_surface_model_from_fit_df(combined)),
        ]:
            summary[name] = dict(
                n=frame.height,
                **dw._prediction_metrics(
                    frame["loss"].to_numpy(),
                    dw._surface_predictions_for_df(fitted, frame),
                ),
            )
        for name, prediction in zip(
            ("random_cv", "compute_cv"), dw._surface_cv_predictions(fit)
        ):
            summary[name] = dw._prediction_metrics(fit["loss"].to_numpy(), prediction)
        boot = _surface_bootstrap_instability(
            data, target_params=1e8, target_compute=1e18
        )
        bounds = np.percentile(boot["optimum_log_ratios"], [2.5, 97.5])
        summary["bootstrap"] = dict(
            n=boot["n_fitted"],
            nonconvex=boot["n_nonconvex"],
            reference_params=1e8,
            reference_compute=1e18,
            log10_ratio_95_interval=bounds.tolist(),
            interval_orders_of_magnitude=float(bounds[1] - bounds[0]),
        )
        summary["early_warmup_counts"] = (
            early.group_by("analysis_training_flops")
            .agg(pl.col("warmup_ongoing").sum())
            .sort("analysis_training_flops")
            .to_dicts()
        )
        report["formulations"][label] = summary
    output = ROOT_DIR / "manuscript/scgpt_dw_rerun_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["formulations"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
