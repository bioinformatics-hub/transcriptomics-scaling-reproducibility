from __future__ import annotations

import numpy as np
import polars as pl

from plotting.geneformer_lr import _quadratic_best_lr_df


LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def _terminal_frame(
    losses: list[float],
    completions: list[float],
    *,
    learning_rates: list[float] = LEARNING_RATES,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target_non_embedding_params": [1_000_000] * len(learning_rates),
            "target_depth": [2] * len(learning_rates),
            "learning_rate": learning_rates,
            "terminal_loss": losses,
            "completion_fraction": completions,
            "backbone_params": [1_000_000.0] * len(learning_rates),
            "non_embedding_params": [1_000_000.0] * len(learning_rates),
            "model.d_model": [256] * len(learning_rates),
            "model.transformer.n_layers": [2] * len(learning_rates),
        }
    )


def test_quadratic_optimum_requires_complete_immediate_neighbors() -> None:
    terminal = _terminal_frame(
        losses=[4.0, 2.0, 1.0, 1.2, 2.0],
        completions=[1.0, 1.0, 1.0, 0.5, 1.0],
    )

    optimum = _quadratic_best_lr_df(terminal, min_completion=0.9).row(
        0, named=True
    )

    assert optimum["lr_fit_method"] == "discrete_incomplete_window"
    assert optimum["optimal_lr"] == 1e-4
    assert optimum["discrete_optimal_lr"] == 1e-4
    assert optimum["n_eligible_lrs"] == 4
    assert np.isnan(optimum["quadratic_a"])


def test_quadratic_optimum_uses_complete_three_point_window() -> None:
    terminal = _terminal_frame(
        losses=[4.0, 2.0, 1.0, 1.4, 3.0],
        completions=[1.0] * 5,
    )

    optimum = _quadratic_best_lr_df(terminal, min_completion=0.9).row(
        0, named=True
    )

    assert optimum["lr_fit_method"] == "quadratic"
    assert 3e-5 < optimum["optimal_lr"] < 3e-4
    assert optimum["n_eligible_lrs"] == 5


def test_missing_sampled_neighbor_forces_discrete_fallback() -> None:
    terminal = _terminal_frame(
        losses=[4.0, 2.0, 1.0, 2.0],
        completions=[1.0] * 4,
        learning_rates=[1e-5, 3e-5, 1e-4, 1e-3],
    )
    second_group = _terminal_frame(
        losses=[5.0, 3.0, 2.0, 3.0, 5.0],
        completions=[1.0] * 5,
    ).with_columns(pl.lit(6).alias("target_depth"))

    optima = _quadratic_best_lr_df(
        pl.concat([terminal, second_group], how="vertical_relaxed"),
        min_completion=0.9,
    )
    optimum = optima.filter(pl.col("target_depth") == 2).row(0, named=True)

    assert optimum["lr_fit_method"] == "discrete_incomplete_window"
    assert optimum["optimal_lr"] == 1e-4
