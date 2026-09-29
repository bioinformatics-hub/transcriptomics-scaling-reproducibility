from collections import Counter
from pathlib import Path
import time

import pandas as pd

from core.config import Pipeline, ScalingConfig
from core.datamodule import CensusDataModule
from tools.utils import load_experiment_and_dataset_from_config
from tools.ontology import (
    SPECIAL,
    build_parents_from_cl_owl,
    build_pruned_tree,
    propagate_leaf_counts,
    visualize_cell_ontology,
)


DEFAULT_K = 30


def collect_validation_leaf_ids(config: ScalingConfig) -> list[str]:
    if config.metadata.pipeline in {Pipeline.DEFAULT, Pipeline.GENEFORMER}:
        return _collect_census_validation_leaf_ids(config)

    datamodule = CensusDataModule(config)
    datamodule.setup()

    leaf_ids: list[str] = []
    for batch in datamodule.val_dataloader():
        if len(batch) not in {6, 7}:
            raise ValueError(
                f"Expected validation batches with 6 or 7 items, got {len(batch)}."
            )
        batch_leaf_ids = batch[5]
        leaf_ids.extend([str(x) for x in batch_leaf_ids])

    if not leaf_ids:
        raise ValueError("Validation dataloader returned no leaf ids.")

    return leaf_ids


def _collect_census_validation_leaf_ids(config: ScalingConfig) -> list[str]:
    """Use the training validation split, reading only observation metadata.

    Label counts determine ontology pruning, so preserve SOMA's seeded split
    rather than independently sampling cells. Expression data and batch order
    are irrelevant to those counts.
    """
    started = time.monotonic()
    print("[coarse_labels] Selecting validation observation IDs", flush=True)
    experiment, dataset = load_experiment_and_dataset_from_config(config)
    try:
        n_cells = len(dataset.query_ids.obs_joinids)
        if not n_cells:
            raise ValueError("Validation dataset is empty.")
        val_perc = 0.05
        if int(n_cells * val_perc) > config.datamodule.max_val_samples:
            val_perc = config.datamodule.max_val_samples / n_cells
        _, validation, _ = dataset.random_split(
            1.0 - 2 * val_perc, val_perc, val_perc, seed=42
        )
        joinids = validation.query_ids.obs_joinids
        print(
            f"[coarse_labels] Selected {len(joinids)} validation cells in "
            f"{time.monotonic() - started:.1f}s; reading labels only",
            flush=True,
        )
        leaf_ids = []
        for offset in range(0, len(joinids), 8192):
            table = experiment.obs.read(
                coords=(joinids[offset:offset + 8192],),
                column_names=["cell_type_ontology_term_id"],
            ).concat()
            leaf_ids.extend(str(value) for value in table.column(0).to_pylist())
        if not leaf_ids:
            raise ValueError("Validation dataset returned no leaf ids.")
        if len(leaf_ids) != len(joinids):
            raise ValueError("Validation metadata row count does not match selected IDs.")
        print(
            f"[coarse_labels] Collected {len(leaf_ids)} labels in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        return leaf_ids
    finally:
        experiment.close()


def save_cell_type_to_coarse_map(
    config_path: str,
    output_csv: str | None = None,
    k: int = DEFAULT_K,
    ontology_path: str = "cl.owl",
    save_visualizations: bool = True,
    skip_if_exists: bool = False,
) -> str:
    config = ScalingConfig(config_path)
    if output_csv is None:
        output_csv = str(
            Path(config.paths.path_to_ckpt_dir) / f"cell_type_to_coarse_k{k}.csv"
        )
    output_path = Path(output_csv)
    if skip_if_exists and output_path.exists():
        mapping = pd.read_csv(output_path, dtype=str)
        required = ["leaf_id", "coarse_id"]
        if not set(required).issubset(mapping.columns) or mapping.empty:
            raise ValueError(f"Invalid coarse-label mapping: {output_path}")
        if (
            mapping[required].isna().any().any()
            or mapping[required].apply(lambda col: col.str.strip().eq("")).any().any()
            or mapping["leaf_id"].duplicated().any()
        ):
            raise ValueError(f"Empty or duplicate IDs in coarse-label mapping: {output_path}")
        print(f"[coarse_labels] Reusing existing mapping: {output_path}", flush=True)
        return str(output_path)

    leaf_ids = collect_validation_leaf_ids(config)
    started = time.monotonic()
    print("[coarse_labels] Parsing ontology", flush=True)
    parents, cl_labels = build_parents_from_cl_owl(ontology_path)
    print(f"[coarse_labels] Parsed ontology in {time.monotonic() - started:.1f}s", flush=True)

    leaf_counts = Counter([leaf for leaf in leaf_ids if leaf not in SPECIAL])
    coarse_ids, final_anchors, current_map = build_pruned_tree(
        leaf_ids=leaf_ids,
        parents=parents,
        k=k,
    )
    print(f"[coarse_labels] Finished pruning after {time.monotonic() - started:.1f}s", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    seen = set()
    for leaf_id in sorted(set(leaf_ids)):
        coarse_id = current_map.get(leaf_id, leaf_id)
        if leaf_id in seen:
            continue
        seen.add(leaf_id)
        mapping_rows.append(
            {
                "leaf_id": leaf_id,
                "leaf_name": cl_labels.get(leaf_id, leaf_id),
                "coarse_id": coarse_id,
                "coarse_name": cl_labels.get(coarse_id, coarse_id),
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(output_path, index=False)

    if save_visualizations:
        output_dir = output_path.parent
        prop_counts = propagate_leaf_counts(leaf_counts, parents)
        visualize_cell_ontology(
            set(leaf_counts.keys()),
            parents,
            cl_labels,
            prop_counts,
            str(output_dir / "01_full_tree"),
            "#D5F5E3",
        )

        pruned_leaf_counts = Counter(
            [anchor for anchor in final_anchors if anchor not in SPECIAL]
        )
        for original_leaf, count in leaf_counts.items():
            target_anchor = current_map[original_leaf]
            pruned_leaf_counts[target_anchor] += count
        pruned_prop_counts = propagate_leaf_counts(pruned_leaf_counts, parents)
        visualize_cell_ontology(
            final_anchors,
            parents,
            cl_labels,
            pruned_prop_counts,
            str(output_dir / "02_pruned_tree"),
            "#AED6F1",
        )

        coarse_counts = Counter(coarse_ids)
        summary_df = pd.DataFrame(
            [
                {
                    "coarse_id": coarse_id,
                    "coarse_name": cl_labels.get(coarse_id, coarse_id),
                    "count": count,
                }
                for coarse_id, count in coarse_counts.most_common()
            ]
        )
        summary_df.to_csv(output_dir / f"coarse_label_counts_k{k}.csv", index=False)

    return str(output_path)


if __name__ == "__main__":
    save_cell_type_to_coarse_map("configs/config_0.yml")
