import yaml
import os
from enum import Enum
from typing import Union
import warnings
import platform


# --- Enums ---
class Pipeline(Enum):
    DEFAULT = "default"
    GENEFORMER = "geneformer"
    GENECORPUS = "genecorpus"
    TOKENIZED_TEXT = "tokenized_text"


class TrainingBehavior(Enum):
    """Historical training semantics, independent of machine-specific paths."""

    LEGACY = "legacy"  # Before the June 18, 2026 loss correction.
    LEGACY_PADDING = "legacy_padding"  # Corrected losses, pre-August 22 masking.
    CORRECTED = "corrected"


def training_behavior(config):
    return getattr(
        getattr(config, "metadata", None),
        "training_behavior",
        TrainingBehavior.CORRECTED,
    )


class ModelClass(Enum):
    FFNN = "ffnn"
    TRANSFORMER = "transformer"
    BIOFORMER = "bioformer"
    NULL = "null"
    TALLY = "tally"
    BERT = "bert"
    BIORANGOTANGO = "biorangotango"


class Label(Enum):
    CELL_TYPE = "cell_type"
    NONE = "none"


class DatasetID(Enum):
    NONE = "none"
    YES = "yes"


class Selection(Enum):
    ACTIVE = "active"
    HVG = "hvg"
    ALL = "all"


class MaskStrategy(Enum):
    UNIFORM = "uniform"
    EQUAL_PROP = "equal_prop"


class SchedulerLR(Enum):
    CONSTANT = "constant"
    ONECYCLE = "onecycle"
    LINEAR = "linear"
    WARMUP_CONSTANT = "warmup_constant"
    WARMUP_CONSTANT_LINEAR = "warmup_constant_linear"


class Precision(Enum):
    FULL = "full"
    MIXED_BF16 = "bf16-mixed"


# --- Utils function ---
def load_yaml(pathto_yaml):
    assert pathto_yaml.endswith(".yml"), f"Expected .yml file, found {pathto_yaml}"
    assert os.path.exists(pathto_yaml), f"Unable to locate {pathto_yaml}."
    try:
        with open(pathto_yaml, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)
    except Exception as e:
        print(f"Error loading {pathto_yaml}:\n\n{e}")
        exit()


# --- Custom Objects ---
class CheckedPath(str):
    def __new__(cls, path, required_ext=None, requires_write=False):
        if path is None:
            warnings.warn("No path provided")
            return None
        if not os.path.exists(path):
            warnings.warn(f"Path does not exist: {path}")
        if required_ext is not None and not path.endswith(required_ext):
            raise ValueError(f"Path must end with {required_ext}")
        if os.path.exists(path) and not os.access(path, mode=os.R_OK):
            raise PermissionError(f"No read permission for {path}")
        if (
            requires_write
            and os.path.exists(path)
            and not os.access(path, mode=os.W_OK)
        ):
            raise PermissionError(f"No write permission for {path}")
        return super().__new__(cls, path)


class CheckedPathW(CheckedPath):
    def __new__(cls, path: str):
        return super().__new__(cls, path, requires_write=True)


class CheckedCSV(CheckedPath):
    def __new__(cls, path: str):
        return super().__new__(cls, path, required_ext=".csv")


class CheckedJSON(CheckedPath):
    def __new__(cls, path: str):
        return super().__new__(cls, path, required_ext=".json")


class CheckedCKPT(CheckedPath):
    def __new__(cls, path: str):
        return super().__new__(cls, path, required_ext=".ckpt")


# --- Update ENUM_MAP ---
ENUM_MAP = {
    "model_class": ModelClass,
    "pipeline": Pipeline,
    "training_behavior": TrainingBehavior,
    "labels": Label,
    "dataset_id": DatasetID,
    "selection": Selection,
    "mask_strategy": MaskStrategy,
    "lr_scheduler": SchedulerLR,
    "path_to_hvg": CheckedCSV,
    "path_to_means": CheckedCSV,
    "path_to_celltypes": CheckedCSV,
    "path_to_dataset_ids": CheckedCSV,
    "path_to_ckpt_file": CheckedCSV,
    "path_to_ckpt_dir": CheckedPathW,
    "path_to_coarse_labels_dir": CheckedPathW,
    "path_to_aimrepo": CheckedPathW,
    "path_to_census": CheckedPath,
    "path_to_finetune_dir": CheckedPath,
    "path_to_genecorpus_dir": CheckedPath,
    "path_to_tokenized_text_dir": CheckedPath,
    "path_to_tokenizer": CheckedJSON,
    "path_to_resume_checkpoint": CheckedPath,
    "precision": Precision,
}


# --- Update TYPE_MAP ---
TYPE_MAP = {
    "run_name": str,
    "user": str,
    "machine": str,
    "total_genes": int,
    "context_length": int,
    "d_model": int,
    "n_layers": int,
    "dropout": float,
    "freeze_encoder": bool,
    "lr": float,
    "obs_value_filter": str,
    "normalize_expr_for_ranking": bool,
    "n_bins": int,
    "mask_pct": float,
    "batch_size": int,
    "io_batch_size": int,
    "val_io_batch_size": int,
    "shuffle_chunk_size": int,
    "use_eager_fetch": bool,
    "checkpoint": bool,
    "checkpoint_steps": int,
    "training_state_checkpoint": bool,
    "training_state_checkpoint_steps": int,
    "resume_training": bool,
    "resume_dataloader": bool,
    "n_epochs": int,
    "n_steps": int,
    "max_time": (str, type(None)),
    "lr_warmup_steps": int,
    "accumulate_grad": int,
    "n_heads": int,
    "d_opm": int,
    "d_z": int,
    "chunk_size": int,
    "gating": bool,
    "norm_first": bool,
    "pair_updates": bool,
    "from_ckpt": bool,
    "title": str,
    "mode": str,
    "target_non_embedding_params": int,
    "actual_non_embedding_params": int,
    "target_depth": int,
    "planned_d_model": int,
    "target_training_flops": float,
    "max_compute_budget": float,
    "sampled_compute_budgets": str,
    "core_max_compute_budget": (float, type(None)),
    "estimated_sec_per_step": float,
    "estimated_gpu_hours": float,
    "max_steps": (int, type(None)),
    "flop_limited_steps": int,
    "forward_flops_per_example": float,
    "training_flops_per_step": float,
    "compute_spearman": bool,
    "val_check_interval": (int, float),
    "check_val_every_n_epoch": (int, type(None)),
    "limit_val_batches": (int, float),
    "find_lr": bool,
    "max_val_samples": int,
    "coarse_k": int,
    "poll_interval_s": int,
    "clustering_method": str,
    "resolution": float,
    "n_neighbors": int,
    "ridge_seed": int,
    "keep_artifacts": bool,
}


class ScalingConfig:
    """
    Implements the attr:value logic in a more controlled way.
    """

    def __init__(self, config_path_or_dict: Union[str, dict] = None, is_nested=False):
        if config_path_or_dict is None:
            config_dict = {}
        elif isinstance(config_path_or_dict, str):
            config_dict = load_yaml(config_path_or_dict)
        elif isinstance(config_path_or_dict, dict):
            config_dict = config_path_or_dict
        else:
            raise TypeError(
                f"config_path_or_dict must be None, str, or dict, found {type(config_path_or_dict)}"
            )

        if isinstance(config_dict.get("metadata", None), dict):
            config_dict["metadata"]["user"] = os.environ.get("USER", None)
            config_dict["metadata"]["machine"] = platform.node()

        self.set_attr_from_dict(config_dict)

        if not is_nested:
            self.check_attributes()

    def set_attr_from_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            self._set_attr(k, v)

    def set_attr_from_dict(self, attr_dict: dict):
        for k, v in attr_dict.items():
            self._set_attr(k, v)

    def _set_attr(self, k, v):
        enum_type = ENUM_MAP.get(k)
        if enum_type:
            # Enum attribute
            v = enum_type(v)
            self.__setattr__(k, v)
        elif k in TYPE_MAP:
            # Simple type attribute
            expected_type = TYPE_MAP[k]
            if not isinstance(v, expected_type):
                raise TypeError(
                    f"Attribute '{k}' must be of type {expected_type}, got {type(v)}."
                )
            self.__setattr__(k, v)
        elif isinstance(v, dict):
            # Nested config: create a new ScalingConfig for this attribute
            nested_config = ScalingConfig(v, is_nested=True)
            self.__setattr__(k, nested_config)
        else:
            raise TypeError(
                f"Attribute '{k}' must be a valid enum, allowed type, or nested dict, got {repr(v)} of type {type(v)} (perhaps unrecognized attribute)."
            )

    def to_dict(self):
        def convert(value):
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, ScalingConfig):
                return value.to_dict()
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            elif isinstance(value, CheckedPath):
                return str(value)
            else:
                return value

        result = {}
        for k, v in self.__dict__.items():
            result[k] = convert(v)
        return result

    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

    def __repr__(self):
        return self.to_yaml()

    def copy(self):
        new = ScalingConfig()
        new.set_attr_from_dict(self.to_dict())
        return new

    def save(self, path_to_yml, overwrite=False):
        assert path_to_yml.endswith(".yml"), (
            f"Expected valid .yml file, found {path_to_yml}."
        )
        if os.path.exists(path_to_yml) and not overwrite:
            raise FileExistsError(
                f"{path_to_yml} already exists and overwrite is False."
            )
        with open(path_to_yml, "w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)

    def check_attributes(self):
        errors = []

        # -- METADATA --
        if hasattr(self, "metadata"):
            if not hasattr(self.metadata, "run_name"):
                errors.append("Metadata must have a 'run_name' attribute.")

        # -- MODEL --
        if hasattr(self, "model"):
            if getattr(self.model, "from_ckpt", False):
                path_to_ckpt = getattr(
                    getattr(self, "paths", None), "path_to_ckpt_file", None
                )
                if not path_to_ckpt:
                    errors.append(
                        "When 'from_ckpt' is True, 'paths.path_to_ckpt_file' must be provided in the config."
                    )
                elif not os.path.exists(str(path_to_ckpt)):
                    errors.append(
                        f"Checkpoint file '{path_to_ckpt}' does not exist. Please provide a valid path."
                    )

            if getattr(self.model, "model_class", None) in {
                ModelClass.TRANSFORMER,
                ModelClass.BIOFORMER,
            }:
                if self.model.model_class == ModelClass.TRANSFORMER:
                    n_heads = self.model.transformer.n_heads
                else:
                    n_heads = self.model.bioformer.n_heads
                d_model = self.model.d_model
                if d_model % n_heads != 0:
                    errors.append(
                        f"'d_model' ({d_model}) must be divisible by 'n_heads' ({n_heads})."
                    )

            if hasattr(self.model, "context_length") and hasattr(
                self.model, "total_genes"
            ):
                if self.model.context_length > self.model.total_genes:
                    errors.append(
                        f"'context_length' ({self.model.context_length}) cannot be greater than 'total_genes' ({self.model.total_genes})."
                    )

        # -- TRAINER --
        if hasattr(self, "trainer"):
            if getattr(self.trainer, "checkpoint", False):
                if not getattr(self.trainer, "checkpoint_steps", False):
                    errors.append(
                        "'checkpoint_steps' must be provided when 'checkpoint' is True in trainer config."
                    )

                if not getattr(self.paths, "path_to_ckpt_dir", False):
                    errors.append(
                        "'path_to_ckpt_dir' must be provided when 'checkpoint' is True in trainer config."
                    )

            if not hasattr(self.trainer, "compute_spearman"):
                errors.append(
                    "'compute_spearman' must be provided in the 'trainer' group"
                )

            if not hasattr(self.trainer, "val_check_interval"):
                errors.append(
                    "'val_check_interval' must be provided in the 'trainer' group"
                )

            if not hasattr(self.trainer, "find_lr"):
                errors.append("'find_lr' must be provided in the 'trainer' group")

            lr_scheduler = getattr(self.trainer, "lr_scheduler", None)
            if (
                isinstance(lr_scheduler, SchedulerLR)
                and lr_scheduler == SchedulerLR.ONECYCLE
            ):
                warnings.warn(
                    "Detected 'onecycle' learning rate scheduler. The 'lr' parameter will be interpreted as 'max_lr'. "
                    "Other scheduler parameters will use their default values.",
                    UserWarning,
                )

        if errors:
            print("\n".join(errors))
            raise ValueError(
                "Config validation failed with the following errors:\n"
                + "\n".join(errors)
            )

    def __getattr__(self, name):
        raise AttributeError(
            f"[ScalingConfig] Attribute '{name}' does not exist. Please check your config.yaml or code for typos or missing fields."
        )
