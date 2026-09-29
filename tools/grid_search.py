import argparse
import os
import copy
import gc
import csv
import math
from .utils import safe_create_dir, safe_copy_dir
from itertools import product
import torch
from lightning.fabric.utilities.throughput import measure_flops

from core.config import ScalingConfig, load_yaml
from core.models import load_model_from_config

TRAINING_FLOPS_MULTIPLIER = 3.0
SWEEP_KEY = "sweep"
PRODUCT_SWEEP_MODE = "product"
PARAMETER_COUNT_DEPTH_SWEEP_MODE = "parameter_count_depth"
EXPLICIT_SWEEP_MODE = "explicit"

# A dictionary defining which parameters are irrelevant for certain model types.
# This helps in avoiding redundant configurations during grid search.
IRRELEVANT_PARAMS = {
    "null": [
        "d_model",
        "n_layers",
        "n_heads",
        "dropout",
        "gating",
        "norm_first",
        "d_z",
        "d_opm",
        "pair_updates",
        "chunk_size",
    ],
    "tally": [
        "d_model",
        "n_layers",
        "n_heads",
        "dropout",
        "gating",
        "norm_first",
        "d_z",
        "d_opm",
        "pair_updates",
        "chunk_size",
    ],
    "ffnn": [
        "n_layers",
        "n_heads",
        "gating",
        "norm_first",
        "d_z",
        "d_opm",
        "pair_updates",
        "chunk_size",
    ],
}

# Each default group of params must be zipped together when defining combinations
# instead of taking all possible cross combinations.
DEFAULT_PARAMS_TO_ZIP = [
    (
        "paths.path_to_census",
        "paths.path_to_hvg",
        "paths.path_to_means",
        "paths.path_to_celltypes",
    ),
]


def get_nested(d, path):
    """Access a nested dictionary value using a dot-separated path."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def set_nested(d, path, value):
    """Set a value in a nested dictionary using a dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def find_grid_params(config_dict, path=""):
    """Recursively find parameters with list values for grid search."""
    grid_params = {}
    for key, value in config_dict.items():
        if path == "" and key == SWEEP_KEY:
            continue
        new_path = f"{path}.{key}" if path else key
        if isinstance(value, list):
            grid_params[new_path] = value
        elif isinstance(value, dict):
            grid_params.update(find_grid_params(value, new_path))
    return grid_params


def get_canonical_key(grid_param_paths, full_config_dict):
    """
    Generates a canonical key for a configuration to handle equivalences.

    This function identifies irrelevant parameters based on the model type
    and other parameter values, allowing the grid search to skip redundant
    combinations. For example, for a 'null' model, parameters like 'd_model'
    or 'n_layers' do not affect the outcome, so all combinations with 'null'
    model but different 'd_model' will map to the same canonical key.

    Args:
        grid_param_paths (list): A list of dot-separated paths for parameters
                                 being varied in the grid search.
        full_config_dict (dict): The complete nested configuration dictionary.

    Returns:
        tuple: A hashable tuple representing the canonical form of the
               configuration, used for deduplication.
    """
    key_parts = {path: get_nested(full_config_dict, path) for path in grid_param_paths}

    model_class = get_nested(full_config_dict, "model.model_class")
    pipeline = get_nested(full_config_dict, "metadata.pipeline")

    # Discard configurations where BERT is used with the default pipeline
    if model_class == "bert" and pipeline == "default":
        return None

    # Prune keys for model-specific sub-configs if the model is not active.
    for path in list(key_parts.keys()):
        if path.startswith("model.transformer") and model_class != "transformer":
            del key_parts[path]
        if path.startswith("model.bioformer") and model_class != "bioformer":
            del key_parts[path]

    if model_class in IRRELEVANT_PARAMS:
        irrelevant = IRRELEVANT_PARAMS[model_class]
        for path in list(key_parts.keys()):
            param_name = path.split(".")[-1]
            if param_name in irrelevant:
                del key_parts[path]

    if model_class == "bioformer":
        # If pair_updates is false, d_z and d_opm are not used.
        pair_updates = get_nested(full_config_dict, "model.bioformer.pair_updates")
        if not pair_updates:
            for path in list(key_parts.keys()):
                param_name = path.split(".")[-1]
                if param_name in ["d_z", "d_opm"]:
                    del key_parts[path]

    if pipeline == "geneformer":
        # If pipeline is geneformer, gene selection is not used.
        for path in list(key_parts.keys()):
            if path.endswith("datamodule.selection"):
                del key_parts[path]

    return tuple(sorted(key_parts.items()))


def get_params_combinations(
    grid_params: dict,
    params_to_zip: list[tuple[str, ...]] | None = None,
) -> tuple[list[str], list[list[str, int, float]]]:
    if params_to_zip is None:
        params_to_zip = DEFAULT_PARAMS_TO_ZIP

    # To store all the params combinations
    all_combinations = []
    final_params = []

    # Cartesian params are those not to be zipped
    params_to_zip_flat = {i for tupl in params_to_zip for i in tupl}
    cartesian_params = [i for i in grid_params if i not in params_to_zip_flat]

    # If they are to be grid searched, then do it
    if cartesian_params:
        cartesian_values = [grid_params[i] for i in cartesian_params]
        cartesian_combinations = list(product(*cartesian_values))
        all_combinations.append(cartesian_combinations)
        final_params.extend(cartesian_params)

    # Now the zipped ones
    for tuple_to_zip in params_to_zip:
        zip_params = [i for i in tuple_to_zip if i in grid_params]
        if zip_params:
            zip_values = [grid_params[i] for i in zip_params]
            zip_lengths = {len(values) for values in zip_values}
            if len(zip_lengths) > 1:
                raise ValueError(
                    "Zipped grid parameters must have equal lengths: "
                    + ", ".join(
                        f"{path}={len(values)}"
                        for path, values in zip(zip_params, zip_values)
                    )
                )
            zip_combination = list(zip(*zip_values))
            all_combinations.append(zip_combination)
            final_params.extend(zip_params)

    # Combine zipped and cartesian
    all_combinations = list(product(*all_combinations))
    all_combinations = [
        [i for tupl in combo for i in tupl] for combo in all_combinations
    ]

    return final_params, all_combinations


def get_sweep_config(config_dict: dict) -> dict:
    sweep_config = config_dict.get(SWEEP_KEY) or {}
    if not isinstance(sweep_config, dict):
        raise TypeError(f"'{SWEEP_KEY}' must be a mapping when provided.")
    return sweep_config


def get_sweep_mode(config_dict: dict) -> str:
    return get_sweep_config(config_dict).get("mode", PRODUCT_SWEEP_MODE)


def get_params_to_zip(config_dict: dict) -> list[tuple[str, ...]]:
    sweep_config = get_sweep_config(config_dict)
    params_to_zip = list(DEFAULT_PARAMS_TO_ZIP)
    extra_zip_params = sweep_config.get("zip_params", [])
    if extra_zip_params is None:
        return params_to_zip
    if not isinstance(extra_zip_params, list):
        raise TypeError("'sweep.zip_params' must be a list of parameter-path lists.")

    for group in extra_zip_params:
        if not isinstance(group, list) or len(group) < 2:
            raise TypeError(
                "Each 'sweep.zip_params' entry must be a list with at least two parameter paths."
            )
        if not all(isinstance(path, str) for path in group):
            raise TypeError("All 'sweep.zip_params' parameter paths must be strings.")
        params_to_zip.append(tuple(group))
    return params_to_zip


def strip_master_only_config(config_dict: dict) -> dict:
    current_config_dict = copy.deepcopy(config_dict)
    current_config_dict.pop(SWEEP_KEY, None)
    return current_config_dict


def load_parameter_depth_lr_table(path: str) -> dict[tuple[int, int], float]:
    lr_table: dict[tuple[int, int], float] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "non_embedding_parameters" not in row:
                raise ValueError(
                    f"LR table '{path}' must include a non_embedding_parameters column."
                )
            target_params = int(float(row["non_embedding_parameters"]))
            for column, value in row.items():
                if not column.startswith("depth_") or value in {None, ""}:
                    continue
                target_depth = int(column.removeprefix("depth_"))
                lr_table[(target_params, target_depth)] = float(value)
    if not lr_table:
        raise ValueError(f"LR table '{path}' did not contain any depth_* values.")
    return lr_table


def get_active_model_config_path(config_dict: dict) -> str:
    model_class = get_nested(config_dict, "model.model_class")
    if model_class in {"transformer", "bert", "biorangotango"}:
        return "model.transformer"
    if model_class == "bioformer":
        return "model.bioformer"
    raise ValueError(
        f"Sweep mode '{PARAMETER_COUNT_DEPTH_SWEEP_MODE}' only supports transformer-like models, got {model_class!r}."
    )


def count_module_parameters(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


def count_non_embedding_parameters(config_dict: dict) -> int:
    config = ScalingConfig(strip_master_only_config(config_dict))
    with torch.device("meta"):
        model = load_model_from_config(config)
        non_embedding_params = count_module_parameters(getattr(model, "encoder", None))
    del model
    gc.collect()
    return non_embedding_params


def estimate_forward_flops_for_config(config_dict: dict) -> float:
    config = ScalingConfig(strip_master_only_config(config_dict))
    with torch.device("meta"):
        model = load_model_from_config(config)
        expr = torch.randint(
            low=0,
            high=config.datamodule.n_bins + 1,
            size=(1, config.model.context_length),
            dtype=torch.float,
        )
        gene_ids = torch.randint(
            low=0,
            high=config.model.total_genes,
            size=(1, config.model.context_length),
            dtype=torch.long,
        )

        def model_fwd(model=model, gene_ids=gene_ids, expr=expr):
            return model(gene_ids, expr)

        forward_flops = float(measure_flops(model, model_fwd))
    del model
    gc.collect()
    return forward_flops


def choose_n_heads(d_model: int, preferred_n_heads: int) -> int:
    if d_model % preferred_n_heads == 0:
        return preferred_n_heads

    candidates = [
        n_heads
        for n_heads in [preferred_n_heads, 16, 12, 8, 6, 4, 3, 2, 1]
        if n_heads > 0 and d_model % n_heads == 0
    ]
    return candidates[0] if candidates else 1


def transformer_backbone_param_estimate(
    d_model: int,
    n_layers: int,
    gating: bool,
) -> int:
    per_layer = 12 * d_model * d_model + 10 * d_model
    if gating:
        per_layer += d_model * d_model + d_model
    return n_layers * per_layer


def solve_d_model_for_target_params(
    base_config_dict: dict,
    target_params: int,
    n_layers: int,
    preferred_n_heads: int,
    d_model_multiple: int,
) -> tuple[int, int, int]:
    model_path = get_active_model_config_path(base_config_dict)
    gating = bool(get_nested(base_config_dict, f"{model_path}.gating"))
    approximate_d_model = int(
        math.sqrt(target_params / (13 if gating else 12) / max(1, n_layers))
    )
    rounded_center = max(
        d_model_multiple,
        int(round(approximate_d_model / d_model_multiple)) * d_model_multiple,
    )
    candidate_d_models = {
        max(d_model_multiple, rounded_center + offset * d_model_multiple)
        for offset in range(-8, 9)
    }
    candidate_d_models.add(d_model_multiple)

    best = None
    for d_model in sorted(candidate_d_models):
        n_heads = choose_n_heads(d_model, preferred_n_heads)
        candidate_config_dict = copy.deepcopy(base_config_dict)
        set_nested(candidate_config_dict, "model.d_model", d_model)
        set_nested(candidate_config_dict, f"{model_path}.n_layers", n_layers)
        set_nested(candidate_config_dict, f"{model_path}.n_heads", n_heads)
        non_embedding_params = count_non_embedding_parameters(candidate_config_dict)
        error = abs(non_embedding_params - target_params)
        if best is None or error < best[0]:
            best = (error, d_model, n_heads, non_embedding_params)

    _, d_model, n_heads, non_embedding_params = best
    return d_model, n_heads, non_embedding_params


def format_run_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def set_run_metadata(config_dict: dict, run_name_parts: list[str]) -> None:
    set_nested(
        config_dict,
        "metadata.title",
        get_nested(config_dict, "metadata.run_name"),
    )
    set_nested(config_dict, "metadata.run_name", "-".join(run_name_parts))


def generate_parameter_count_depth_configs(
    base_config: dict,
    output_folder: str,
    config_count: int,
) -> int:
    sweep_config = get_sweep_config(base_config)
    target_params_list = sweep_config["target_non_embedding_params"]
    target_depths = sweep_config["target_depths"]
    target_training_flops = float(sweep_config["target_training_flops"])
    max_steps = sweep_config.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    checkpoint_at_end = bool(sweep_config.get("checkpoint_at_end", False))
    lr_table = None
    if sweep_config.get("lr_table"):
        lr_table = load_parameter_depth_lr_table(str(sweep_config["lr_table"]))
    d_model_multiple = int(sweep_config.get("d_model_multiple", 4))
    model_path = get_active_model_config_path(base_config)
    preferred_n_heads = int(
        sweep_config.get(
            "n_heads",
            get_nested(base_config, f"{model_path}.n_heads"),
        )
    )

    stripped_base_config = strip_master_only_config(base_config)
    grid_params = find_grid_params(stripped_base_config)
    param_names, all_combinations = get_params_combinations(
        grid_params,
        params_to_zip=get_params_to_zip(base_config),
    )
    if not all_combinations:
        all_combinations = [[]]

    print(
        "Generating parameter-count/depth sweep for "
        f"{len(target_params_list)} parameter targets, {len(target_depths)} depths, "
        f"and {len(all_combinations)} remaining grid combinations."
    )

    architecture_cache = {}
    config_count_start = config_count
    for combination in all_combinations:
        grid_config_dict = copy.deepcopy(stripped_base_config)
        run_name_parts = []
        for path, value in zip(param_names, combination):
            set_nested(grid_config_dict, path, value)
            short_name = (
                path.replace(".", "_")
                .replace("metadata_", "")
                .replace("model_", "")
                .replace("datamodule_", "")
                .replace("trainer_", "")
            )
            run_name_parts.append(f"{short_name}_{format_run_value(value)}")

        for target_params, n_layers in product(target_params_list, target_depths):
            target_params = int(target_params)
            n_layers = int(n_layers)
            architecture_param_paths = [
                path for path in param_names if path not in {"model.lr"}
            ]
            architecture_key = (
                target_params,
                n_layers,
                preferred_n_heads,
                d_model_multiple,
                tuple(
                    sorted(
                        (path, get_nested(grid_config_dict, path))
                        for path in architecture_param_paths
                    )
                ),
            )

            if architecture_key not in architecture_cache:
                d_model, n_heads, non_embedding_params = (
                    solve_d_model_for_target_params(
                        grid_config_dict,
                        target_params=target_params,
                        n_layers=n_layers,
                        preferred_n_heads=preferred_n_heads,
                        d_model_multiple=d_model_multiple,
                    )
                )
                architecture_config_dict = copy.deepcopy(grid_config_dict)
                set_nested(architecture_config_dict, "model.d_model", d_model)
                set_nested(architecture_config_dict, f"{model_path}.n_layers", n_layers)
                set_nested(architecture_config_dict, f"{model_path}.n_heads", n_heads)
                forward_flops = estimate_forward_flops_for_config(
                    architecture_config_dict
                )
                training_flops_per_step = (
                    forward_flops
                    * TRAINING_FLOPS_MULTIPLIER
                    * get_nested(architecture_config_dict, "datamodule.batch_size")
                    * get_nested(architecture_config_dict, "trainer.accumulate_grad")
                )
                flop_limited_steps = max(
                    1, math.ceil(target_training_flops / training_flops_per_step)
                )
                n_steps = (
                    min(max_steps, flop_limited_steps)
                    if max_steps is not None
                    else flop_limited_steps
                )
                architecture_cache[architecture_key] = (
                    d_model,
                    n_heads,
                    non_embedding_params,
                    forward_flops,
                    training_flops_per_step,
                    flop_limited_steps,
                    n_steps,
                )

            (
                d_model,
                n_heads,
                non_embedding_params,
                forward_flops,
                training_flops_per_step,
                flop_limited_steps,
                n_steps,
            ) = architecture_cache[architecture_key]

            current_config_dict = copy.deepcopy(grid_config_dict)
            set_nested(current_config_dict, "model.d_model", d_model)
            set_nested(current_config_dict, f"{model_path}.n_layers", n_layers)
            set_nested(current_config_dict, f"{model_path}.n_heads", n_heads)
            if lr_table is not None:
                lr_key = (target_params, n_layers)
                if lr_key not in lr_table:
                    raise ValueError(
                        "Missing LR-table value for "
                        f"target_non_embedding_params={target_params}, depth={n_layers}."
                    )
                set_nested(current_config_dict, "model.lr", lr_table[lr_key])
            set_nested(current_config_dict, "trainer.n_steps", n_steps)
            if sweep_config.get("validate_at_end", False):
                set_nested(current_config_dict, "trainer.val_check_interval", n_steps)
                set_nested(current_config_dict, "trainer.check_val_every_n_epoch", None)
            if checkpoint_at_end and get_nested(current_config_dict, "trainer.checkpoint"):
                set_nested(current_config_dict, "trainer.checkpoint_steps", n_steps)
            current_config_dict["sweep_metadata"] = {
                "mode": PARAMETER_COUNT_DEPTH_SWEEP_MODE,
                "target_non_embedding_params": target_params,
                "actual_non_embedding_params": non_embedding_params,
                "target_depth": n_layers,
                "target_training_flops": target_training_flops,
                "max_steps": max_steps,
                "flop_limited_steps": flop_limited_steps,
                "forward_flops_per_example": forward_flops,
                "training_flops_per_step": training_flops_per_step,
            }
            set_run_metadata(
                current_config_dict,
                run_name_parts
                + [
                    f"target_params_{target_params:g}",
                    f"depth_{n_layers:g}",
                    f"d_model_{d_model:g}",
                    f"n_heads_{n_heads:g}",
                ],
            )

            config_obj = ScalingConfig(current_config_dict)
            output_file_name = os.path.join(output_folder, f"config_{config_count}.yml")
            config_obj.save(output_file_name, overwrite=True)
            print(f"  Generated: {output_file_name}")
            config_count += 1

    print(
        f"\nGenerated {config_count - config_count_start} unique configuration files."
    )
    print("Configuration file generation complete!")
    return config_count


def generate_explicit_configs(
    base_config: dict,
    output_folder: str,
    config_count: int,
) -> int:
    sweep_config = get_sweep_config(base_config)
    runs = sweep_config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("'sweep.runs' must be a non-empty list for explicit sweep mode.")

    stripped_base_config = strip_master_only_config(base_config)
    config_count_start = config_count
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("Each explicit sweep run must be a mapping.")
        run_name = run.get("run_name")
        overrides = run.get("overrides", {})
        if not isinstance(run_name, str) or not run_name:
            raise ValueError("Each explicit sweep run must provide a non-empty run_name.")
        if not isinstance(overrides, dict):
            raise TypeError("Each explicit sweep run's overrides must be a mapping.")

        current_config_dict = copy.deepcopy(stripped_base_config)
        set_run_metadata(current_config_dict, [run_name])
        for path, value in overrides.items():
            set_nested(current_config_dict, path, value)
        if (
            "trainer.checkpoint_steps" in overrides
            and "trainer.training_state_checkpoint_steps" not in overrides
            and get_nested(current_config_dict, "trainer.training_state_checkpoint")
        ):
            set_nested(
                current_config_dict,
                "trainer.training_state_checkpoint_steps",
                overrides["trainer.checkpoint_steps"],
            )

        metadata = run.get("sweep_metadata", {})
        if metadata:
            if not isinstance(metadata, dict):
                raise TypeError("Each explicit sweep run's sweep_metadata must be a mapping.")
            current_config_dict["sweep_metadata"] = metadata

        config_obj = ScalingConfig(current_config_dict)
        output_file_name = os.path.join(output_folder, f"config_{config_count}.yml")
        config_obj.save(output_file_name, overwrite=True)
        print(f"  Generated: {output_file_name}")
        config_count += 1

    print(
        f"\nGenerated {config_count - config_count_start} explicit configuration files."
    )
    print("Configuration file generation complete!")
    return config_count


def generate_grid_configs(
    config_file="config.yml", output_folder="configs", config_count=0, overwrite=True
):
    """
    Generates multiple configuration files based on a grid search approach
    for parameters with list values in the input config file.

    Args:
        config_file (str): The path to the base configuration YAML file.
        output_folder (str): The name of the subfolder to save generated configs.
    """
    if not os.path.exists(config_file):
        print(f"Error: The file '{config_file}' was not found.")
        return

    safe_create_dir(output_folder, overwrite=overwrite)

    # Load the base config as a raw dictionary first to handle lists for grid search
    base_config = load_yaml(config_file)
    sweep_mode = get_sweep_mode(base_config)
    if sweep_mode == PARAMETER_COUNT_DEPTH_SWEEP_MODE:
        return generate_parameter_count_depth_configs(
            base_config=base_config,
            output_folder=output_folder,
            config_count=config_count,
        )
    if sweep_mode == EXPLICIT_SWEEP_MODE:
        return generate_explicit_configs(
            base_config=base_config,
            output_folder=output_folder,
            config_count=config_count,
        )
    if sweep_mode != PRODUCT_SWEEP_MODE:
        raise ValueError(f"Unknown sweep mode: {sweep_mode}")
    params_to_zip = get_params_to_zip(base_config)
    base_config = strip_master_only_config(base_config)
    grid_params = find_grid_params(base_config)

    if not grid_params:
        print(
            "No list parameters found for grid search. A single config file will be generated."
        )
        # Since there are no lists, the file should be a valid config.
        # We can now safely instantiate ScalingConfig and save it.
        try:
            config_obj = ScalingConfig(base_config)
            output_file_path = os.path.join(output_folder, f"config_{config_count}.yml")
            config_obj.save(output_file_path, overwrite=True)
            print(f"Single config file generated: {output_file_path}")
        except Exception as e:
            print(
                f"The provided config file '{config_file}' has no lists for grid search, but it is not a valid configuration."
            )
            print(f"Error: {e}")
        return

    # Generate all combinations of grid parameters
    param_names, all_combinations = get_params_combinations(
        grid_params,
        params_to_zip=params_to_zip,
    )

    print(f"Found {len(grid_params)} parameters to vary: {', '.join(param_names)}")
    print(
        f"Analyzing {len(all_combinations)} total combinations to generate unique configs in '{output_folder}'..."
    )

    # Create and save config files for each unique combination
    seen_canonical_keys = set()
    config_count_start = config_count
    for combination in all_combinations:
        # Create a full config dict for this combination by starting with a copy
        # of the base config and updating it with the current combination.
        current_config_dict = copy.deepcopy(base_config)
        for path, value in zip(param_names, combination):
            set_nested(current_config_dict, path, value)

        canonical_key = get_canonical_key(param_names, current_config_dict)

        # Skip invalid configurations
        if canonical_key is None or canonical_key in seen_canonical_keys:
            continue

        seen_canonical_keys.add(canonical_key)

        # Generate a descriptive run name from the varying parameters
        run_name_parts = []
        for path, value in zip(param_names, combination):
            # Use a short, clean name for the parameter in the run name
            short_name = (
                path.replace(".", "_")
                .replace("metadata_", "")
                .replace("model_", "")
                .replace("datamodule_", "")
            )
            run_name_parts.append(f"{short_name}_{value}")
        run_name = "-".join(run_name_parts)
        # set run name in base config as title in runs
        set_run_metadata(current_config_dict, [run_name])

        # Instantiate a config object to handle validation and saving
        config_obj = ScalingConfig(current_config_dict)
        output_file_name = os.path.join(output_folder, f"config_{config_count}.yml")
        config_obj.save(output_file_name, overwrite=True)
        print(f"  Generated: {output_file_name}")
        config_count += 1

    print(
        f"\nGenerated {config_count - config_count_start} unique configuration files."
    )
    print("Configuration file generation complete!")
    return config_count


def generate_grid_configs_master(master_folder, output_folder, old_master: str = None):
    """
    Generates grid search configurations for all YAML files in a master folder,
    skipping redundant combinations across multiple config files.

    Args:
        master_folder (str): The path to the master folder containing YAML files.
        output_folder (str): The name of the subfolder to save generated configs.
    """
    config_count = 0
    if not os.path.exists(master_folder):
        print(f"Error: The folder '{master_folder}' was not found.")
        return
    if old_master:
        safe_copy_dir(master_folder, old_master)
    safe_create_dir(output_folder)
    print(
        f"Generating grid search configurations from all YAML files in '{master_folder}'..."
    )
    for filename in os.listdir(master_folder):
        if filename.endswith(".yml"):
            assert (
                filename.split(".")[0]
                == load_yaml(os.path.join(master_folder, filename))["metadata"][
                    "run_name"
                ]
            ), f"Run name in {filename} does not match the filename. Please fix it."
            # Process each YAML file in the master folder
            config_file = os.path.join(master_folder, filename)
            print(f"Processing config file: {config_file}")
            count = generate_grid_configs(
                config_file=config_file,
                output_folder=output_folder,
                config_count=config_count,
                overwrite=False,
            )
            if count is not None:
                config_count = count
    print(f"\nTotal unique configuration files generated: {config_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configuration files for a grid search, skipping redundant combinations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="config.yml",
        help="Path to the base YAML configuration file (default: config.yml).",
    )
    parser.add_argument(
        "--master", type=str, default="", help="Path to the master config folder"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="configs",
        help="Directory to save the generated config files (default: configs).",
    )
    parser.add_argument(
        "--old_master",
        type=str,
        default="old_master",
        help="Directory with older master configs",
    )
    args = parser.parse_args()

    if args.master != "":
        generate_grid_configs_master(
            master_folder=args.master,
            output_folder=args.output,
            old_master=args.old_master,
        )
    else:
        generate_grid_configs(config_file=args.config, output_folder=args.output)
