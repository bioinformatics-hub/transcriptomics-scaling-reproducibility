import lightning as pl
import torch
from typing import Union, Tuple, Iterator
import tiledbsoma_ml as soma_ml
from attrs import evolve
from torch.utils.data import DataLoader, IterableDataset
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from tools.hvg import subset_genes
from core.config import (
    ScalingConfig,
    Label,
    MaskStrategy,
    Selection,
    Pipeline,
    TrainingBehavior,
    training_behavior,
)
from tools.utils import (
    build_dataset_id_map,
    load_experiment_and_dataset_from_config,
    validate_total_genes_against_census,
)
import os
import warnings
from collections import deque
from itertools import batched
from pathlib import Path


def _load_dataset_id_map_from_csv(
    path_to_dataset_ids: str | None,
) -> dict[str, int] | None:
    if not path_to_dataset_ids:
        return None
    if not os.path.exists(path_to_dataset_ids):
        warnings.warn(
            f"Dataset-id cache not found at {path_to_dataset_ids}; falling back to Census scan.",
            UserWarning,
        )
        return None

    dataset_ids_df = pd.read_csv(path_to_dataset_ids)
    if "dataset_id" not in dataset_ids_df:
        warnings.warn(
            f"Dataset-id cache {path_to_dataset_ids} has no 'dataset_id' column; falling back to Census scan.",
            UserWarning,
        )
        return None

    dataset_ids = sorted(
        str(dataset_id)
        for dataset_id in dataset_ids_df["dataset_id"].dropna().unique().tolist()
    )
    if not dataset_ids:
        warnings.warn(
            f"Dataset-id cache {path_to_dataset_ids} is empty; falling back to Census scan.",
            UserWarning,
        )
        return None

    print(
        f"[CensusDataModule] Loaded {len(dataset_ids)} dataset ids from {path_to_dataset_ids}"
    )
    return {dataset_id: i for i, dataset_id in enumerate(dataset_ids)}


def _get_dataset_id_map(config: ScalingConfig, experiment) -> dict[str, int]:
    dataset_id_map = _load_dataset_id_map_from_csv(
        getattr(config.paths, "path_to_dataset_ids", None)
    )
    if dataset_id_map is not None:
        return dataset_id_map

    print(
        "[CensusDataModule] Building dataset-id map from Census obs; "
        "this can be slow on large shared-storage datasets."
    )
    return build_dataset_id_map(
        experiment,
        obs_value_filter=config.datamodule.obs_value_filter,
    )


def _n_cells_from_query_ids(experiment_ds: soma_ml.ExperimentDataset) -> int | None:
    query_ids = getattr(experiment_ds, "query_ids", None)
    obs_joinids = getattr(query_ids, "obs_joinids", None)
    if obs_joinids is None:
        return None
    try:
        return len(obs_joinids)
    except TypeError:
        return None


def _configured_val_io_batch_size(config: ScalingConfig) -> int | None:
    env_value = os.environ.get("SCALING_SOMA_VAL_IO_BATCH_SIZE")
    if env_value is not None:
        return int(env_value)
    value = getattr(config.datamodule, "val_io_batch_size", None)
    if value is not None:
        return int(value)
    return None


def _fast_forward_experiment_dataset(
    dataset: soma_ml.ExperimentDataset,
    consumed_batches: int,
) -> soma_ml.ExperimentDataset:
    """Return a dataset positioned after committed batches without reading them.

    ``ExperimentDataset`` applies two deterministic shuffles: one to chunks of
    observation IDs and another within each I/O batch. Reproduce both shuffles
    using IDs alone, discard the committed prefix, and disable further
    shuffling on the resulting dataset. This avoids fetching and transforming
    every previously committed cell on each resumed Slurm segment.
    """
    if consumed_batches <= 0:
        return dataset
    if int(getattr(dataset, "world_size", 1)) != 1:
        raise NotImplementedError(
            "Fast cursor resume currently requires a single training process"
        )

    query_ids = dataset.query_ids
    obs_joinids = query_ids.obs_joinids
    rows_to_skip = min(
        int(consumed_batches) * int(dataset.batch_size), len(obs_joinids)
    )

    if dataset.shuffle:
        chunks = query_ids.shuffle_chunks(
            shuffle_chunk_size=dataset.shuffle_chunk_size,
            seed=dataset.seed,
        )
        io_batches = batched(
            (joinid for chunk in chunks for joinid in chunk),
            dataset.io_batch_size,
        )
        io_shuffle_rng = np.random.default_rng(dataset.seed)
        remaining_batches: list[np.ndarray] = []
        for obs_coords in io_batches:
            ordered_coords = io_shuffle_rng.permuted(obs_coords)
            if rows_to_skip >= len(ordered_coords):
                rows_to_skip -= len(ordered_coords)
                continue
            if rows_to_skip:
                ordered_coords = ordered_coords[rows_to_skip:]
                rows_to_skip = 0
            remaining_batches.append(ordered_coords)
    else:
        remaining_batches = [obs_joinids[rows_to_skip:]]

    remaining_joinids = (
        np.concatenate(remaining_batches)
        if remaining_batches
        else np.empty(0, dtype=obs_joinids.dtype)
    )
    remaining_query_ids = evolve(query_ids, obs_joinids=remaining_joinids)
    return evolve(
        dataset,
        query_ids=remaining_query_ids,
        shuffle=False,
        use_eager_fetch=dataset.use_eager_fetch,
    )


class TrainingDataCursor:
    def __init__(self, config: ScalingConfig):
        run_dir = (
            Path(config.paths.path_to_ckpt_dir)
            / config.metadata.title
            / config.metadata.run_name
        )
        self.state_path = run_dir / "training_state" / "data_cursor.json"
        self.consumed_batches = 0
        self.consumed_cells = 0
        self.completed_epochs = 0
        self.last_batch_soma_joinids: list[int] = []
        self._seen_batches_this_iter = 0
        self._pending_batches: deque[list[int]] = deque()
        if getattr(config.trainer, "resume_training", False):
            self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        with open(self.state_path) as f:
            state = json.load(f)
        self.consumed_batches = int(state.get("consumed_batches", 0))
        self.consumed_cells = int(state.get("consumed_cells", 0))
        self.completed_epochs = int(state.get("completed_epochs", 0))
        self.last_batch_soma_joinids = [
            int(value) for value in state.get("last_batch_soma_joinids", [])
        ]

    def state_dict(self) -> dict:
        return {
            "consumed_batches": self.consumed_batches,
            "consumed_cells": self.consumed_cells,
            "completed_epochs": self.completed_epochs,
            "last_batch_soma_joinids": self.last_batch_soma_joinids,
        }

    def load_state_dict(self, state: dict) -> None:
        self.consumed_batches = int(state.get("consumed_batches", 0))
        self.consumed_cells = int(state.get("consumed_cells", 0))
        self.completed_epochs = int(state.get("completed_epochs", 0))
        self.last_batch_soma_joinids = [
            int(value) for value in state.get("last_batch_soma_joinids", [])
        ]
        self._seen_batches_this_iter = 0
        self._pending_batches.clear()

    def begin_iteration(self, total_batches: int) -> None:
        """Start an epoch, advancing a cursor that finished the prior epoch."""

        if total_batches <= 0:
            self._seen_batches_this_iter = 0
            return
        if self.consumed_batches >= total_batches:
            completed, remaining = divmod(self.consumed_batches, total_batches)
            self.completed_epochs += completed
            self.consumed_batches = remaining
            self.last_batch_soma_joinids = []
        self._seen_batches_this_iter = 0

    def mark_committed_prefix_as_positioned(self) -> None:
        """Tell the cursor that the dataset already removed its saved prefix."""
        self._seen_batches_this_iter = self.consumed_batches

    def save_resume_marker(
        self, *, checkpoint_path: str | Path, global_step: int
    ) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        state = self.state_dict()
        state["checkpoint_path"] = str(checkpoint_path)
        state["global_step"] = int(global_step)
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, self.state_path)

    def should_skip_next_batch(self) -> bool:
        if self._seen_batches_this_iter < self.consumed_batches:
            self._seen_batches_this_iter += 1
            return True
        return False

    def is_exhausted(self, total_batches: int) -> bool:
        return self.consumed_batches >= total_batches

    def stage_batch(self, soma_joinids: list[int]) -> None:
        self._seen_batches_this_iter += 1
        self._pending_batches.append(soma_joinids)

    def commit_next_batch(self) -> None:
        if not self._pending_batches:
            raise RuntimeError(
                "Training completed a batch but the resumable dataloader has no "
                "corresponding pending batch."
            )
        soma_joinids = self._pending_batches.popleft()
        self.consumed_batches += 1
        self.consumed_cells += len(soma_joinids)
        self.last_batch_soma_joinids = soma_joinids

    def commit_pending_batches(self) -> None:
        while self._pending_batches:
            self.commit_next_batch()


class ScalingDataset(IterableDataset):
    def __init__(
        self,
        dataset: soma_ml.ExperimentDataset,
        config: ScalingConfig,
        dataset_id_map: dict[str, int],
        data_cursor: TrainingDataCursor | None = None,
    ):
        super().__init__()

        self.dataset = dataset
        self._legacy_masking = training_behavior(config) != TrainingBehavior.CORRECTED
        self.data_cursor = data_cursor
        self.labels = config.datamodule.labels
        self.mask_pct = config.datamodule.mask_pct
        self.total_genes = config.model.total_genes
        self.context_length = config.model.context_length
        self.dataset_id_map = dataset_id_map
        self.dataset_timing_batches = int(
            os.environ.get("SCALING_DATASET_TIMING_BATCHES", "0")
        )
        self.path_to_celltypes = getattr(config.paths, "path_to_celltypes", None)
        self.path_to_census = getattr(config.paths, "path_to_census", None)
        self.obs_value_filter = getattr(config.datamodule, "obs_value_filter", None)

        # Get cell types map
        if self.labels == Label.CELL_TYPE:
            cell_types = pd.read_csv(self.path_to_celltypes)
            self.cell_type_map = {
                ct: i for i, ct in enumerate(cell_types["cell_type"].unique())
            }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Same behaviour as in tiledbsoma_ml.ExperimentDataset, copied from there
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )

    def get_y_batch(self, obs_batch):
        if self.labels == Label.NONE:
            y_batch = None
        elif self.labels == Label.CELL_TYPE:
            # Map cell_type string to integer label
            try:
                y_batch = np.array(
                    [self.cell_type_map[ct] for ct in obs_batch.cell_type]
                )
            except KeyError as exc:
                missing = str(exc.args[0])
                raise KeyError(
                    "Encountered a cell_type that is absent from the cached "
                    f"celltypes CSV: {missing!r}. Regenerate metadata caches for "
                    "the exact Census path and obs filter used by this run. "
                    f"path_to_celltypes={self.path_to_celltypes!r}, "
                    f"path_to_census={self.path_to_census!r}, "
                    f"obs_value_filter={self.obs_value_filter!r}."
                ) from exc
            y_batch = torch.tensor(y_batch, dtype=torch.long)
        else:
            raise NotImplementedError()
        return y_batch

    def get_dataset_id(self, obs_batch):
        # Map dataset string to integer label
        y_dataset = np.array([self.dataset_id_map[di] for di in obs_batch.dataset_id])
        y_dataset = torch.tensor(y_dataset, dtype=torch.long)
        return y_dataset

    def get_leaf_id_batch(self, obs_batch):
        leaf_ids = obs_batch.cell_type_ontology_term_id
        return ["unknown" if pd.isna(x) else str(x) for x in leaf_ids]

    def _batch_soma_joinids(self, obs_batch) -> list[int]:
        if hasattr(obs_batch, "__getitem__") and "soma_joinid" in obs_batch:
            values = obs_batch["soma_joinid"]
        else:
            values = obs_batch.soma_joinid
        return [int(value) for value in values]

    def _should_skip_batch(self) -> bool:
        if self.data_cursor is None:
            return False
        if self.data_cursor.should_skip_next_batch():
            return True
        return False

    def _stage_batch(self, obs_batch) -> None:
        if self.data_cursor is not None:
            self.data_cursor.stage_batch(self._batch_soma_joinids(obs_batch))

    def _iter_dataset_batches(self):
        if self.data_cursor is not None:
            self.data_cursor.begin_iteration(len(self.dataset))
        batch_index = 0
        iteration_dataset = self.dataset
        if self.data_cursor is not None and self.data_cursor.consumed_batches:
            fast_forward_start = time.perf_counter()
            iteration_dataset = _fast_forward_experiment_dataset(
                self.dataset,
                self.data_cursor.consumed_batches,
            )
            print(
                "[CensusDataModule] Fast-forwarded training input past "
                f"{self.data_cursor.consumed_batches} committed batches; "
                f"{len(iteration_dataset)} batches remain "
                f"({time.perf_counter() - fast_forward_start:.1f}s)",
                flush=True,
            )
            self.data_cursor.mark_committed_prefix_as_positioned()
        dataset_iter = iter(iteration_dataset)
        while True:
            raw_fetch_start = time.perf_counter()
            try:
                batch = next(dataset_iter)
            except StopIteration:
                return
            raw_fetch_s = time.perf_counter() - raw_fetch_start
            if batch_index < self.dataset_timing_batches:
                x_batch, _ = batch
                print(
                    f"[DatasetTiming] {type(self).__name__} raw_fetch batch={batch_index} "
                    f"seconds={raw_fetch_s:.2f} x_shape={getattr(x_batch, 'shape', None)}",
                    flush=True,
                )
            yield batch_index, batch
            batch_index += 1

    def _log_transform_timing(self, batch_index: int, transform_s: float) -> None:
        if batch_index < self.dataset_timing_batches:
            print(
                f"[DatasetTiming] {type(self).__name__} transform batch={batch_index} "
                f"seconds={transform_s:.2f}",
                flush=True,
            )


class BinnedMaskedDataset(ScalingDataset):
    """Wrapper Class for TileDB-SOMA-ML's ExperimentDataset

    Every batch emitted by it is first transformed then passed downstream to the DataLoader.
    Since TileDB-SOMA-ML handles the sampling and shuffling we have to avoid using those
    functionalities from the PyTorch DataLoader

    This class handles the Binning of the expression levels, the masking,
    and the selection of the genes passed to the model,
    as well as the (optional) creation of labels.
    """

    def __init__(
        self,
        dataset: soma_ml.ExperimentDataset,
        config: ScalingConfig,
        dataset_id_map: dict[str, int],
        data_cursor: TrainingDataCursor | None = None,
    ):
        super().__init__(dataset, config, dataset_id_map, data_cursor)

        self.selection = config.datamodule.selection
        self.mask_strategy = config.datamodule.mask_strategy
        self.n_bins = config.datamodule.n_bins

        if self.selection == Selection.HVG:
            df = pd.read_csv(config.paths.path_to_hvg)
            df["highly_variable"] = subset_genes(
                mean=df["means"],
                dispersion_norm=df["dispersions_norm"].to_numpy(),
                n_top_genes=self.context_length,
            )
            self.hvg = df[df["highly_variable"]].index

    def mask(self, x: np.ndarray, gene_ids: np.ndarray) -> np.ndarray:
        """Choose expression tokens to mask, excluding padded positions."""
        valid_tokens = gene_ids < self.total_genes

        if self.mask_strategy == MaskStrategy.UNIFORM:
            masked_tokens = (np.random.rand(*x.shape) < self.mask_pct) & valid_tokens
        elif self.mask_strategy == MaskStrategy.EQUAL_PROP:
            # Aim for half zero and half nonzero targets. If one group is too
            # small, allocate the remaining probability mass to the other.
            zero_tokens = valid_tokens & (x == 0)
            active_tokens = valid_tokens & (x > 0)
            num_tokens = int(valid_tokens.sum())
            num_zeros = int(zero_tokens.sum())
            num_active = int(active_tokens.sum())
            target = self.mask_pct * num_tokens

            target_active = min(target / 2, num_active)
            target_zeros = min(target - target_active, num_zeros)
            target_active = min(target - target_zeros, num_active)

            probs = np.zeros(x.shape, dtype=float)
            if num_active:
                probs[active_tokens] = target_active / num_active
            if num_zeros:
                probs[zero_tokens] = target_zeros / num_zeros
            masked_tokens = (np.random.rand(*x.shape) < probs) & valid_tokens
        else:
            raise NotImplementedError()

        return masked_tokens

    def mask_old(self, x: np.ndarray) -> np.ndarray:
        """Legacy masking used by runs produced before the padding-mask fix."""
        if self.mask_strategy == MaskStrategy.UNIFORM:
            masked_tokens = np.random.rand(*x.shape) <= self.mask_pct
        elif self.mask_strategy == MaskStrategy.EQUAL_PROP:
            tokens = (x < self.total_genes).sum()
            goal = 0.5 * self.mask_pct * tokens
            num_zeros = (x == 0).sum()
            num_ones = tokens - num_zeros
            p_ones = min(goal, num_ones) / num_ones
            p_zeros = (self.mask_pct * tokens - min(goal, num_ones)) / num_zeros
            probs = np.where(x == 0, p_zeros, p_ones)
            masked_tokens = np.random.rand(*x.shape) <= probs
            masked_tokens[x >= self.total_genes] = False
        else:
            raise NotImplementedError()

        return masked_tokens

    def selection_sizes(self) -> tuple[int, int]:
        """Return active/inactive slots while preserving the context length."""
        inactive = int(self.context_length * (self.mask_pct / 2))
        active = self.context_length - inactive
        return active, inactive

    def selection_sizes_old(self) -> tuple[int, int]:
        """Legacy independently-rounded active/inactive slot counts."""
        active = int(self.context_length * (1 - self.mask_pct / 2))
        inactive = int(self.context_length * (self.mask_pct / 2))
        return active, inactive

    def __iter__(
        self,
    ):
        return self._iter_batches(legacy_masking=self._legacy_masking)

    def iter_old(self):
        """Legacy iterator with the historical padding ID and masking path."""
        return self._iter_batches(legacy_masking=True)

    def _iter_batches(
        self,
        legacy_masking: bool,
    ) -> Iterator[
        Tuple[
            torch.IntTensor,
            torch.IntTensor,
            torch.BoolTensor,
            Union[None, torch.IntTensor],
            Union[None, torch.IntTensor],
        ]
    ]:
        for batch_index, (x_batch, obs_batch) in self._iter_dataset_batches():
            transform_start = time.perf_counter()
            if self.data_cursor is not None and self.data_cursor.is_exhausted(
                len(self.dataset)
            ):
                break
            if self._should_skip_batch():
                continue
            num_genes = x_batch.shape[1]
            # --------- GENE IDS ---------
            gene_batch = np.broadcast_to(
                np.arange(num_genes, dtype=int), shape=x_batch.shape
            )

            # -------- BINNING PROCEDURE ----------
            x_batch[x_batch == 0] = (
                np.nan
            )  # we replace zeros with nans so we can use nanquantile funcion from numpy
            qs = np.linspace(
                0, 1, self.n_bins
            )  # we create the quantiles, evenly spaced so we'll get equal frequency bins
            # nanquantile is a (BINS, NUM_CELLS) array, we change it to (BINS, NUM_CELLS, 1)
            # then we compare arr (NUM_CELLS, GENES) to it leveraging broadcasting, it will prepend a 1 dimension to arr
            # we'll get a matrix with True if arr element is in that quantile
            # we sum up the True occurrences across quantiles, we'll get the number corresponding to the bin the value belongs to
            # note that if the value is 0 it will belong to bin 0, there are BINS additional bins
            x_batch = (x_batch <= np.nanquantile(x_batch, qs, axis=1)[..., None]).sum(
                axis=0
            )

            # --------- SELECTION PROCEDURE ---------
            if self.selection == Selection.ALL:
                pass
            elif self.selection == Selection.ACTIVE:
                # we want to select either active or masked genes
                active, inactive = (
                    self.selection_sizes_old()
                    if legacy_masking
                    else self.selection_sizes()
                )
                selected_genes = np.zeros(
                    shape=(x_batch.shape[0], active + inactive), dtype=int
                )
                selected_expr = np.zeros(
                    shape=(x_batch.shape[0], active + inactive), dtype=int
                )

                for i in range(x_batch.shape[0]):
                    candidate_genes = gene_batch[i, x_batch[i, :] > 0]
                    candidate_zeros = gene_batch[i, x_batch[i, :] == 0]
                    np.random.shuffle(candidate_genes)
                    np.random.shuffle(candidate_zeros)
                    chosen = candidate_genes[:active]
                    selected_genes[i, : len(chosen)] = chosen
                    selected_genes[i, len(chosen) : active] = (
                        self.total_genes + 1 if legacy_masking else self.total_genes
                    )  # PAD token
                    selected_genes[i, active:] = candidate_zeros[:inactive]
                    selected_expr[i, : len(chosen)] = x_batch[
                        i, chosen
                    ]  # the rest can comfortably stay zero

                x_batch = selected_expr
                gene_batch = selected_genes

            elif self.selection == Selection.HVG:
                x_batch = x_batch[:, self.hvg]
                gene_batch = gene_batch[:, self.hvg]
            else:
                raise NotImplementedError()

            # --------- MASKING ---------
            mask_batch = (
                self.mask_old(x_batch)
                if legacy_masking
                else self.mask(x_batch, gene_batch)
            )

            # --------- LABELS (optional) ---------
            y_batch = self.get_y_batch(obs_batch)
            y_dataset = self.get_dataset_id(obs_batch)

            # --------- TENSOR CREATION ---------
            # using the from_numpy method so no new memory is employed
            # then Lightning will copy them to VRAM when loading them
            # x_batch, gene_batch, mask_batch = map(torch.from_numpy, [x_batch, gene_batch, mask_batch])
            # may be a problem since numpy by default makes them non writbale i guess, will go back to this later
            x_batch, gene_batch, mask_batch = map(
                torch.tensor, [x_batch, gene_batch, mask_batch]
            )

            # Will use nn.Linear for encoding bins (expects float)
            x_batch = x_batch.float()

            leaf_id_batch = self.get_leaf_id_batch(obs_batch)
            self._log_transform_timing(
                batch_index, time.perf_counter() - transform_start
            )
            self._stage_batch(obs_batch)
            yield x_batch, gene_batch, mask_batch, y_batch, y_dataset, leaf_id_batch


class RankValueDataset(ScalingDataset):
    """Wrapper Class for TileDB-SOMA-ML's ExperimentDataset

    Every batch emitted by it is first transformed then passed downstream to the DataLoader.
    Since TileDB-SOMA-ML handles the sampling and shuffling we have to avoid using those
    functionalities from the PyTorch DataLoader

    This class handles the rank value encoding of the expression levels, the masking,
    and the selection of the genes passed to the model,
    as well as the (optional) creation of labels.
    """

    def __init__(
        self,
        dataset: soma_ml.ExperimentDataset,
        config: ScalingConfig,
        dataset_id_map: dict[str, int],
        data_cursor: TrainingDataCursor | None = None,
    ):
        super().__init__(dataset, config, dataset_id_map, data_cursor)

        self.normalize_expr_for_ranking = config.datamodule.normalize_expr_for_ranking

        try:
            # Load in the means
            self.expr_means = pd.read_csv(config.paths.path_to_means)[
                "means"
            ].to_numpy()
            # to avoid genes never seen expressed shoot up to infinity (nan)
            # set avg expr to 1 artifically
            self.expr_means[np.isclose(self.expr_means, 0)] = 1.0
        except Exception:
            print("Error loading means, skipping normalization step.")
            self.expr_means = np.ones(self.total_genes)

    def get_global_gene_ranks(self, expr_row):
        """
        Rank all genes by their normalized expression values.

        Returns an array mapping gene_id -> global rank, where rank 0 is the
        most highly expressed gene in the cell.
        """

        # Normalize by average expression if required
        ranking_expr = expr_row
        if self.normalize_expr_for_ranking:
            ranking_expr = expr_row / self.expr_means[: expr_row.shape[0]]

        # Sort by normalized expression, descending.
        sorted_ids = ranking_expr.argsort()[::-1]
        global_ranks = np.empty(expr_row.shape[0], dtype=int)
        global_ranks[sorted_ids] = np.arange(expr_row.shape[0], dtype=int)

        return global_ranks

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            torch.IntTensor,
            torch.IntTensor,
            torch.BoolTensor,
            Union[None, torch.IntTensor],
            Union[None, torch.IntTensor],
        ]
    ]:
        # x_batch is (B, G) tensor of expression values (all genes)
        for batch_index, (x_batch, obs_batch) in self._iter_dataset_batches():
            transform_start = time.perf_counter()
            if self.data_cursor is not None and self.data_cursor.is_exhausted(
                len(self.dataset)
            ):
                break
            if self._should_skip_batch():
                continue
            # tiledb soma dataset returns array
            assert isinstance(x_batch, np.ndarray)

            # How many genes in this batch
            num_genes = x_batch.shape[1]

            # normalize counts across cells
            x_batch = x_batch / x_batch.sum(axis=1, keepdims=True)

            # Create list of gene tokens (just the arange)
            gene_batch = np.broadcast_to(
                np.arange(num_genes, dtype=int), shape=x_batch.shape
            )

            # To store the selectd ones
            selected_genes = np.zeros(
                shape=(x_batch.shape[0], self.context_length), dtype=int
            )

            # To store thir expression
            selected_expr = np.zeros(
                shape=(x_batch.shape[0], self.context_length), dtype=float
            )

            # to store the mask
            mask_batch = np.zeros_like(selected_genes, dtype=bool)

            # To store the original ranks of the selected genes
            position_batch = np.full_like(selected_genes, fill_value=-1, dtype=int)

            # For each batch, select the active ones
            # and if too many, randoly subset
            for i in range(x_batch.shape[0]):
                # Get id of active genes
                active_ids = x_batch[i, :].nonzero()[0]
                global_ranks = self.get_global_gene_ranks(x_batch[i, :])

                # Shuffle if too many
                if len(active_ids) > self.context_length:
                    np.random.shuffle(active_ids)

                # chosen
                chosen_ids = active_ids[: self.context_length]
                chosen_ranks = global_ranks[chosen_ids]
                rank_order = chosen_ranks.argsort()
                chosen_genes = gene_batch[i, chosen_ids][rank_order]
                chosen_ranks = chosen_ranks[rank_order]

                # Masking (we do by row and for non-pad position,
                # to ensure even distribution)
                assert len(chosen_genes.shape) == 1
                mask_ids = np.random.rand(len(chosen_genes)) <= self.mask_pct
                mask_batch[i, : len(mask_ids)] = mask_ids

                # Now add to the buffer and pad if needed
                selected_genes[i, : len(chosen_genes)] = chosen_genes
                position_batch[i, : len(chosen_ranks)] = chosen_ranks

                # Pad if needed
                selected_genes[i, len(chosen_genes) :] = (
                    self.total_genes
                )  # PAD token (not +1 since genes are from 0 to total_genes - 1)

                # Store their expression
                selected_expr[i, : len(chosen_genes)] = x_batch[
                    i, chosen_genes
                ]  # the rest can comfortably stay zero

            # The batch to yield
            gene_batch = selected_genes

            # no expression in this pipeline
            # x_batch = np.zeros_like(selected_genes)
            # expression levels for the genes selected
            x_batch = selected_expr

            # --------- LABELS (optional) ---------
            y_batch = self.get_y_batch(obs_batch)
            y_dataset = self.get_dataset_id(obs_batch)

            # --------- TENSOR CREATION ---------
            # using the from_numpy method so no new memory is employed
            # then Lightning will copy them to VRAM when loading them
            # x_batch, gene_batch, mask_batch = map(torch.from_numpy, [x_batch, gene_batch, mask_batch])
            # may be a problem since numpy by default makes them non writbale i guess, will go back to this later
            x_batch, gene_batch, mask_batch, position_batch = map(
                torch.tensor, [x_batch, gene_batch, mask_batch, position_batch]
            )

            # Will use nn.Linear for encoding bins (expects float)
            x_batch = x_batch.float()

            leaf_id_batch = self.get_leaf_id_batch(obs_batch)
            self._log_transform_timing(
                batch_index, time.perf_counter() - transform_start
            )
            self._stage_batch(obs_batch)
            yield (
                x_batch,
                gene_batch,
                mask_batch,
                y_batch,
                y_dataset,
                leaf_id_batch,
                position_batch,
            )


class ArrowDataset(IterableDataset):
    """Wrapper Class for Arrow Dataset"""

    def __init__(self, config: ScalingConfig, split: str):
        super().__init__()

        self.split = split
        self.labels = config.datamodule.labels
        self.mask_pct = config.datamodule.mask_pct
        self.total_genes = config.model.total_genes
        self.context_length = config.model.context_length
        self.batch_size = config.datamodule.batch_size

        # Get cell types map
        if self.labels == Label.CELL_TYPE:
            cell_types = pd.read_csv(config.paths.path_to_celltypes)
            self.cell_type_map = {
                ct: i for i, ct in enumerate(cell_types["cell_type"].unique())
            }

        # TODO fix this
        if config.metadata.pipeline == Pipeline.GENECORPUS:
            base_dir = config.paths.path_to_genecorpus_dir
        elif config.metadata.pipeline == Pipeline.TOKENIZED_TEXT:
            base_dir = config.paths.path_to_tokenized_text_dir
        if self.split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split {self.split}, must be one of ['train', 'val', 'test']"
            )
        self.size = 150_000 if self.split == "train" else 1_500
        self.filepath = os.path.join(base_dir, f"{self.split}.arrow")

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            torch.IntTensor,
            torch.IntTensor,
            torch.BoolTensor,
            Union[None, torch.IntTensor],
        ]
    ]:
        rows_emitted = 0
        current_chunk_buffer = []
        # Open the Arrow IPC file

        with pa.memory_map(self.filepath, "r") as source:
            reader = ipc.open_stream(source)

            while rows_emitted < self.size:
                try:
                    batch = reader.read_next_batch()
                    if batch is None:  # No more batches in this file
                        break

                    # Determine how many rows from this batch we still need
                    rows_needed_from_total = self.size - rows_emitted
                    rows_to_process_in_batch = min(
                        batch.num_rows, rows_needed_from_total
                    )

                    if rows_to_process_in_batch == 0:
                        break  # No more rows needed or available

                    # Slice the batch if we only need a portion of it
                    if rows_to_process_in_batch < batch.num_rows:
                        batch = batch.slice(0, rows_to_process_in_batch)

                    # Convert the Arrow RecordBatch to a Pandas DataFrame for easy row iteration
                    # This might load the whole *batch* into memory temporarily.
                    batch_df = batch.to_pandas()

                    for _, row_series in batch_df.iterrows():
                        if rows_emitted >= self.size:
                            break  # Stop adding if we've reached total limit

                        current_chunk_buffer.append(row_series.to_dict()["input_ids"])
                        rows_emitted += 1

                        if len(current_chunk_buffer) == self.batch_size:
                            # Convert the current chunk buffer to a tensor
                            padded_chunk_buffer = []
                            for el in current_chunk_buffer:
                                padded = (
                                    np.ones(shape=(2048,), dtype=int) * self.pad_token
                                )  # PAD token
                                padded[: len(el)] = el
                                padded_chunk_buffer.append(padded)
                            gene_batch = torch.tensor(
                                np.vstack(padded_chunk_buffer), dtype=torch.long
                            )
                            gene_batch = gene_batch[:, : self.context_length]
                            mask_batch = torch.zeros_like(gene_batch, dtype=torch.bool)
                            for i in range(gene_batch.shape[0]):
                                # Masking (we do by row and for non-pad position,
                                # to ensure even distribution)
                                mask_ids = (
                                    torch.rand(gene_batch.shape[1]) <= self.mask_pct
                                ) & (gene_batch[i] != self.pad_token)
                                mask_batch[i, : len(mask_ids)] = mask_ids

                            x_batch = self.create_x_batch(
                                self.batch_size, self.context_length
                            )
                            yield x_batch, gene_batch, mask_batch, None
                            current_chunk_buffer = []  # Reset buffer

                except StopIteration:
                    break  # No more batches available
        # If there are any remaining rows in the buffer, they are skipped for simplicity

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index: int):
        # Same behaviour as in tiledbsoma_ml.ExperimentDataset, copied from there
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )


class GenecorpusDataset(ArrowDataset):
    """Wrapper Class for Genecorpus Dataset"""

    def __init__(self, config: ScalingConfig, split: str):
        super().__init__(config, split)
        self.pad_token = self.total_genes

    def create_x_batch(self, batch_size, sequence_length):
        # since it has been shown that genes expression follows a Zipf distribution
        # we create a tensor that follows the same distribution
        # and recover the normalized gene expression based on gene ranks alone
        x_batch = self.create_zipf_tensor_pytorch(self.batch_size, self.context_length)
        return x_batch

    def create_zipf_tensor_pytorch(self, batch_size, sequence_length):
        """
        Creates a tensor where elements sum up to one along the sequence_length dimension
        and are progressively lower in a Zipf-like fashion, using PyTorch.

        Args:
            batch_size: The desired batch size.
            sequence_length: The desired sequence length.

        Returns:
            A PyTorch tensor of shape (batch_size, sequence_length)
            with the specified properties.
        """

        # 1. Generate a Zipf-like sequence (1/k)
        # Use torch.arange to get indices from 1 to sequence_length
        # Ensure it's a float tensor for division
        indices = torch.arange(1, sequence_length + 1, dtype=torch.float32)
        zipf_sequence = 1.0 / indices

        # 2. Normalize the sequence so it sums to one
        normalized_zipf_sequence = zipf_sequence / torch.sum(zipf_sequence)

        # 3. Broadcast the tensor to (batch_size, sequence_length)
        # Use unsqueeze(0) to add a batch dimension (making it 1xsequence_length)
        # Then use expand(batch_size, -1) to repeat it across the batch dimension.
        # The -1 tells expand to infer the size of that dimension from the original tensor.
        broadcasted_tensor = normalized_zipf_sequence.unsqueeze(0).expand(
            batch_size, -1
        )

        return broadcasted_tensor


class TokenizedTextDataset(ArrowDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.pad_token = 0

    def create_x_batch(self, batch_size, sequence_length):
        x_batch = torch.zeros(size=(batch_size, sequence_length), dtype=torch.float32)
        return x_batch


class CensusDataModule(pl.LightningDataModule):
    """DataModule responsible for opening the Census (only human data), filtering,
    creating a TileDB-SOMA-ML Dataset, performing the splits, then wrapping it up with our custom class,
    and create the various train, val, test DataLoaders
    """

    def __init__(self, config: ScalingConfig):
        super().__init__()

        # Required by datamodule
        self.pipeline = config.metadata.pipeline
        self.path_to_census = config.paths.path_to_census
        self.obs_value_filter = config.datamodule.obs_value_filter
        self.batch_size = config.datamodule.batch_size

        # To be passed to datasets
        self.config = config

        # Fail before Lightning initializes CUDA. A too-small gene vocabulary can
        # otherwise turn valid Census feature IDs into device-side assertions.
        if self.pipeline in {Pipeline.DEFAULT, Pipeline.GENEFORMER}:
            validate_total_genes_against_census(config)

    def setup(self, stage=None):
        if (
            self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            dataset_class = (
                TokenizedTextDataset
                if self.pipeline == Pipeline.TOKENIZED_TEXT
                else GenecorpusDataset
            )
            self.train_dataset = dataset_class(self.config, split="train")
            self.val_dataset = dataset_class(self.config, split="val")
            self.test_dataset = dataset_class(self.config, split="test")
            return

        setup_start = time.perf_counter()
        experiment, experiment_ds = load_experiment_and_dataset_from_config(self.config)
        print(
            f"[CensusDataModule] Opened SOMA dataset in {time.perf_counter() - setup_start:.1f}s"
        )
        dataset_id_start = time.perf_counter()
        dataset_id_map = _get_dataset_id_map(self.config, experiment)
        print(
            f"[CensusDataModule] Prepared dataset-id map in {time.perf_counter() - dataset_id_start:.1f}s"
        )

        # Reduce test size if too big
        # (Soma TileDB does not support passing
        # ints as split params, but only percs)
        n_cells = _n_cells_from_query_ids(experiment_ds)
        if n_cells is None:
            n_batches, _ = experiment_ds.shape
            n_cells = n_batches * self.config.datamodule.batch_size
        else:
            n_batches = int(np.ceil(n_cells / self.config.datamodule.batch_size))
        val_perc = 0.05
        val_samples = int(n_cells * val_perc)
        print(
            f"(Datamodule Size Check) n_batches: {n_batches} | batch_size: {self.config.datamodule.batch_size} | n_cells: {n_cells}"
        )
        if val_samples > self.config.datamodule.max_val_samples:
            val_perc = self.config.datamodule.max_val_samples / n_cells
            warnings.warn(
                f"Number of val samples ({val_samples}) exceeds max_val_samples ({self.config.datamodule.max_val_samples}), decreased 'val_perc' to {val_perc}."
            )
        train_dataset, val_dataset, test_dataset = experiment_ds.random_split(
            1.0 - 2 * val_perc, val_perc, val_perc, seed=42
        )
        val_io_batch_size = _configured_val_io_batch_size(self.config)
        if val_io_batch_size is not None:
            val_dataset = evolve(val_dataset, io_batch_size=val_io_batch_size)
            test_dataset = evolve(test_dataset, io_batch_size=val_io_batch_size)
            print(
                "[CensusDataModule] Using validation/test SOMA io_batch_size="
                f"{val_io_batch_size}"
            )

        # Pipeline-specific iterator
        if self.pipeline == Pipeline.DEFAULT:
            dataset_class = BinnedMaskedDataset
        elif self.pipeline == Pipeline.GENEFORMER:
            dataset_class = RankValueDataset

        # Create the datasets with the correct pipeline
        train_cursor = None
        if getattr(self.config.trainer, "resume_dataloader", False):
            train_cursor = TrainingDataCursor(self.config)

        self.train_dataset = dataset_class(
            train_dataset, self.config, dataset_id_map, train_cursor
        )
        self.val_dataset = dataset_class(val_dataset, self.config, dataset_id_map)
        self.test_dataset = dataset_class(test_dataset, self.config, dataset_id_map)

        return

    def train_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.test_dataset)
