"""Read a local CELLxGENE Census snapshot's manuscript provenance fields.

The script performs metadata-only reads, except for the small RNA feature table.
It does not read the expression matrix or modify the Census snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiledbsoma as soma
import tiledbsoma_ml as soma_ml


DEFAULT_CENSUS_PATH = Path("data/census/census_data/homo_sapiens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report Census release metadata, filtered cell and split counts, "
            "and an ordered RNA feature-vocabulary checksum."
        )
    )
    parser.add_argument(
        "--census-path",
        type=Path,
        default=DEFAULT_CENSUS_PATH,
        help="Path to the homo_sapiens SOMA Experiment.",
    )
    parser.add_argument(
        "--release",
        default="2025-11-08",
        help="Known Census release identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--download-date",
        help=(
            "Known local download date in YYYY-MM-DD form. This cannot be "
            "reliably inferred from filesystem timestamps."
        ),
    )
    parser.add_argument(
        "--obs-filter",
        default="is_primary_data == True",
        help="Observation value filter used in training.",
    )
    parser.add_argument(
        "--layer",
        default="normalized",
        help="RNA expression layer used in training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Physical batch size used to construct ExperimentDataset.",
    )
    parser.add_argument(
        "--max-validation-cells",
        type=int,
        default=30_000,
        help="Cap applied independently to validation and test splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print JSON to stdout.",
    )
    parser.add_argument(
        "--feature-vocabulary-output",
        type=Path,
        help="Optional TSV output path for the ordered Ensembl vocabulary.",
    )
    return parser.parse_args()


def utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    return str(value)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dataset_cell_count(dataset: soma_ml.ExperimentDataset) -> int:
    return len(dataset.query_ids.obs_joinids)


def ordered_feature_vocabulary(experiment: soma.Experiment) -> list[tuple[int, str]]:
    features = (
        experiment.ms["RNA"]
        .var.read(column_names=["soma_joinid", "feature_id"])
        .concat()
        .to_pandas()
        .sort_values("soma_joinid")
    )
    return [
        (int(joinid), str(feature_id))
        for joinid, feature_id in features[["soma_joinid", "feature_id"]].itertuples(
            index=False, name=None
        )
    ]


def vocabulary_sha256(vocabulary: list[tuple[int, str]]) -> str:
    digest = hashlib.sha256()
    for joinid, feature_id in vocabulary:
        digest.update(f"{joinid}\t{feature_id}\n".encode())
    return digest.hexdigest()


def write_vocabulary(path: Path, vocabulary: list[tuple[int, str]]) -> None:
    lines = ["soma_joinid\tfeature_id\n"]
    lines.extend(f"{joinid}\t{feature_id}\n" for joinid, feature_id in vocabulary)
    path.write_text("".join(lines), encoding="utf-8")


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    census_path = args.census_path.resolve()
    path_stat = census_path.stat()

    with soma.open(census_path.as_posix()) as experiment:
        metadata = {
            str(key): json_value(value) for key, value in experiment.metadata.items()
        }
        vocabulary = ordered_feature_vocabulary(experiment)

        with experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=args.obs_filter),
        ) as query:
            filtered_cells = int(query.n_obs)
            dataset = soma_ml.ExperimentDataset(
                query,
                layer_name=args.layer,
                batch_size=args.batch_size,
                io_batch_size=65_536,
                shuffle=True,
                shuffle_chunk_size=64,
                seed=42,
                use_eager_fetch=True,
            )

        validation_fraction = 0.05
        if int(filtered_cells * validation_fraction) > args.max_validation_cells:
            validation_fraction = args.max_validation_cells / filtered_cells
        train, validation, test = dataset.random_split(
            1.0 - 2 * validation_fraction,
            validation_fraction,
            validation_fraction,
            seed=42,
        )

    if args.feature_vocabulary_output:
        write_vocabulary(args.feature_vocabulary_output, vocabulary)

    return {
        "census": {
            "release_identifier": args.release,
            "download_date": args.download_date,
            "path": census_path.as_posix(),
            "path_name_contains_known_typo": "2015-11-08" in census_path.as_posix(),
            "path_mtime_utc_not_a_download_date": utc_timestamp(path_stat.st_mtime),
            "experiment_metadata": metadata,
        },
        "selection": {
            "organism": "homo_sapiens",
            "measurement": "RNA",
            "expression_layer": args.layer,
            "observation_filter": args.obs_filter,
            "post_filter_cells": filtered_cells,
        },
        "partitions": {
            "seed": 42,
            "validation_and_test_fraction_before_cap": 0.05,
            "validation_and_test_cap": args.max_validation_cells,
            "effective_validation_and_test_fraction": validation_fraction,
            "training_cells": dataset_cell_count(train),
            "validation_cells": dataset_cell_count(validation),
            "test_cells": dataset_cell_count(test),
        },
        "feature_vocabulary": {
            "count": len(vocabulary),
            "ordering": "ascending soma_joinid",
            "sha256_of_tab_separated_joinid_and_feature_id_rows": (
                vocabulary_sha256(vocabulary)
            ),
            "first_entry": vocabulary[0] if vocabulary else None,
            "last_entry": vocabulary[-1] if vocabulary else None,
            "output_path": (
                args.feature_vocabulary_output.resolve().as_posix()
                if args.feature_vocabulary_output
                else None
            ),
        },
        "software": {
            "tiledbsoma": package_version("tiledbsoma"),
            "tiledbsoma_ml": package_version("tiledbsoma-ml"),
        },
    }


def main() -> None:
    args = parse_args()
    report = collect_provenance(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
