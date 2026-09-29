import os
import shutil
import argparse
import warnings
import logging
import pandas as pd

import tiledbsoma as soma
import tiledbsoma_ml as soma_ml
# import cellxgene_census

from core.config import ScalingConfig

TILEDB_CONFIG = {
    "py.init_buffer_bytes": 1 * 1024**3,
    "soma.init_buffer_bytes": 1 * 1024**3,
}

_TILEDB_CONCURRENCY_ENV = {
    "SCALING_TILEDB_COMPUTE_CONCURRENCY": "sm.compute_concurrency_level",
    "SCALING_TILEDB_IO_CONCURRENCY": "sm.io_concurrency_level",
}


def soma_tiledb_context() -> soma.options.SOMATileDBContext:
    """Create a SOMA context with optional process-local thread limits."""
    tiledb_config: dict[str, int] = {}
    for env_name, config_name in _TILEDB_CONCURRENCY_ENV.items():
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            concurrency = int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"{env_name} must be a positive integer, got {env_value!r}"
            ) from exc
        if concurrency < 1:
            raise ValueError(
                f"{env_name} must be a positive integer, got {env_value!r}"
            )
        tiledb_config[config_name] = concurrency

    return soma.options.SOMATileDBContext(tiledb_config=tiledb_config or None)


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def safe_copy_dir(source_dir, destination_dir):
    """
    Copies all files from a source directory to a destination directory,
    handling naming conflicts by appending a copy number.

    Args:
        source_dir (str): The path to the source directory.
        destination_dir (str): The path to the destination directory.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Destination directory '{destination_dir}' created.")

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)

        if os.path.isfile(source_path):
            destination_path = os.path.join(destination_dir, filename)

            if os.path.exists(destination_path):
                name, extension = os.path.splitext(filename)
                copy_num = 1
                while True:
                    new_filename = f"{name}_copy_{copy_num}{extension}"
                    new_destination_path = os.path.join(destination_dir, new_filename)
                    if not os.path.exists(new_destination_path):
                        shutil.copy2(source_path, new_destination_path)
                        print(f"Conflict for '{filename}': copied as '{new_filename}'.")
                        break
                    copy_num += 1
            else:
                shutil.copy2(source_path, destination_path)
                print(f"Copied '{filename}' to '{destination_dir}'.")


def safe_create_dir(dir, overwrite=True):
    # Check and potentially clear the output dir
    if os.path.exists(dir):
        if os.listdir(dir):  # Check if dir is not empty
            print(f"The dir '{dir}' already exists and is not empty.")
            if overwrite:
                print("I'm deleting , hopefully you knew what you were doing.")
                try:
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                    print(f"Contents of '{dir}' deleted.")
                except OSError as e:
                    print(f"Error deleting dir '{dir}': {e}")
                    return
    else:
        os.makedirs(dir)


def parse_and_load_config() -> ScalingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="?", default="local_config.yml")
    args = parser.parse_args()
    return ScalingConfig(args.config)


def suppress_messages():
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def load_experiment_and_dataset_from_config(
    config: ScalingConfig,
    batch_size_override: int | None = None,
    shuffle: bool = True,
):
    def env_or_config_int(env_name: str, attr_name: str, default: int) -> int:
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return int(env_value)
        return int(getattr(config.datamodule, attr_name, default))

    def env_or_config_bool(env_name: str, attr_name: str, default: bool) -> bool:
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(getattr(config.datamodule, attr_name, default))

    experiment, experiment_ds = load_experiment_and_dataset(
        path_to_experiment=config.paths.path_to_census,
        batch_size=(
            batch_size_override
            if batch_size_override is not None
            else config.datamodule.batch_size
        ),
        obs_value_filter=config.datamodule.obs_value_filter,
        shuffle=shuffle,
        io_batch_size=env_or_config_int("SCALING_SOMA_IO_BATCH_SIZE", "io_batch_size", 65536),
        shuffle_chunk_size=env_or_config_int(
            "SCALING_SOMA_SHUFFLE_CHUNK_SIZE", "shuffle_chunk_size", 64
        ),
        use_eager_fetch=env_or_config_bool(
            "SCALING_SOMA_EAGER_FETCH", "use_eager_fetch", True
        ),
    )
    try:
        validate_total_genes_against_experiment(config, experiment)
    except Exception:
        experiment.close()
        raise
    return experiment, experiment_ds


def validate_total_genes_against_experiment(
    config: ScalingConfig,
    experiment,
) -> int:
    """Validate the configured gene vocabulary against Census metadata."""
    configured_num_genes = int(config.model.total_genes)
    observed_num_genes = experiment.ms["RNA"].var.count
    if callable(observed_num_genes):
        observed_num_genes = observed_num_genes()
    observed_num_genes = int(observed_num_genes)

    if configured_num_genes != observed_num_genes:
        raise ValueError(
            "Invalid model.total_genes: configured "
            f"{configured_num_genes}, but the Census at "
            f"{config.paths.path_to_census!r} contains {observed_num_genes} RNA "
            "features. Set model.total_genes to the Census feature count before "
            "training. A mismatch can create gene token IDs outside the model "
            "vocabulary and surface as a CUDA device-side assertion."
        )

    return observed_num_genes


def validate_total_genes_against_census(config: ScalingConfig) -> int:
    """Open only Census metadata and validate ``model.total_genes``."""
    with soma.open(
        uri=config.paths.path_to_census,
        context=soma_tiledb_context(),
    ) as experiment:
        return validate_total_genes_against_experiment(config, experiment)


def load_experiment_and_dataset(
    path_to_experiment: str,
    batch_size: int = 4,
    obs_value_filter: str = "is_primary_data == True",
    layer_name: str = "normalized",
    shuffle: bool = True,
    io_batch_size: int = 65536,
    shuffle_chunk_size: int = 64,
    use_eager_fetch: bool = True,
):
    layer_uri = os.path.join(path_to_experiment, "ms", "RNA", "X", layer_name)
    assert os.path.exists(layer_uri), f"Cannot locate: {layer_uri}"

    experiment = soma.open(
        uri=path_to_experiment,
        # context=cellxgene_census.get_default_soma_context()
        context=soma_tiledb_context(),  # substituting for cellxgene_census which breaks all the envs
    )

    with experiment.axis_query(
        measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    ) as query:
        experiment_ds = soma_ml.ExperimentDataset(
            query,
            layer_name=layer_name,
            obs_column_names=[
                "soma_joinid",
                "cell_type",
                "dataset_id",
                "cell_type_ontology_term_id",
            ],
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle=shuffle,
            shuffle_chunk_size=shuffle_chunk_size,
            seed=42,
            use_eager_fetch=use_eager_fetch,
        )

    return experiment, experiment_ds


def build_dataset_id_map(
    experiment,
    obs_value_filter: str = "is_primary_data == True",
) -> dict[str, int]:
    obs_df = experiment.obs.read(
        column_names=["dataset_id"],
        value_filter=obs_value_filter,
    ).concat().to_pandas()
    dataset_ids = sorted(
        str(dataset_id)
        for dataset_id in pd.Series(obs_df["dataset_id"]).dropna().unique().tolist()
    )
    return {dataset_id: i for i, dataset_id in enumerate(dataset_ids)}


def format_seconds_dhms(seconds):
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days:02}:{hours:02}:{minutes:02}:{secs:02}"
