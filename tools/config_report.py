import gc
import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch

from core.config import Precision, ScalingConfig
from core.models import load_model_from_config
from tools.grid_search import generate_grid_configs, generate_grid_configs_master
from lightning.fabric.utilities.throughput import measure_flops


PARAM_DTYPE_BYTES = 4
ADAM_STATE_BYTES_PER_PARAM = 8
TRAINING_FLOPS_MULTIPLIER = 3.0
MIN_INFERENCE_RUNTIME_OVERHEAD_BYTES = 256 * 1024**2
MIN_TRAIN_RUNTIME_OVERHEAD_BYTES = 512 * 1024**2
INFERENCE_RUNTIME_OVERHEAD_PCT = 0.05
TRAIN_RUNTIME_OVERHEAD_PCT = 0.10


def count_parameters(model) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def get_activation_dtype_bytes(config: ScalingConfig) -> int:
    trainer_precision = getattr(config.trainer, "precision", Precision.FULL)
    if trainer_precision == Precision.MIXED_BF16:
        return 2
    return 4


def get_model_depth_stats(config: ScalingConfig) -> tuple[int, int | None]:
    model_class = config.model.model_class.value
    if model_class in {"transformer", "bert", "biorangotango"}:
        return config.model.transformer.n_layers, config.model.transformer.n_heads
    if model_class == "bioformer":
        return config.model.bioformer.n_layers, config.model.bioformer.n_heads
    if model_class == "ffnn":
        return 1, None
    return 0, None


def estimate_encoder_activation_memory_bytes(config: ScalingConfig) -> tuple[int, int]:
    batch_size = config.datamodule.batch_size
    context_length = config.model.context_length
    d_model = config.model.d_model
    n_layers, n_heads = get_model_depth_stats(config)
    act_bytes = get_activation_dtype_bytes(config)
    model_class = config.model.model_class.value

    token_activations = batch_size * context_length * d_model * act_bytes
    if model_class in {"null", "tally"}:
        return token_activations, 0

    # Training needs to retain more than the visible hidden states: residual paths,
    # QKV projections, MLP intermediates, and normalization/dropout inputs.
    hidden_factor = (
        16
        if model_class in {"transformer", "bioformer", "bert", "biorangotango"}
        else 6
    )
    hidden_bytes = hidden_factor * max(1, n_layers) * token_activations

    attention_bytes = 0
    if n_heads is not None:
        attention_bytes = (
            6
            * batch_size
            * n_heads
            * context_length
            * context_length
            * max(1, n_layers)
            * act_bytes
        )

    pair_bytes = 0
    if model_class == "bioformer" and config.model.bioformer.pair_updates:
        pair_bytes = (
            batch_size
            * context_length
            * context_length
            * config.model.bioformer.d_z
            * act_bytes
        )

    return hidden_bytes + attention_bytes + pair_bytes, pair_bytes


def estimate_decoder_memory_bytes(config: ScalingConfig) -> tuple[int, int, int]:
    batch_size = config.datamodule.batch_size
    context_length = config.model.context_length
    act_bytes = get_activation_dtype_bytes(config)
    pipeline = config.metadata.pipeline.value
    model_class = config.model.model_class.value
    mask_pct = getattr(config.datamodule, "mask_pct", 0.0)

    if (
        pipeline in {"geneformer", "genecorpus", "tokenized_text"}
        or model_class == "bert"
    ):
        vocab_size = config.model.total_genes + 2
        logits_bytes = batch_size * context_length * vocab_size * act_bytes
        masked_logits_bytes = int(logits_bytes * mask_pct)
        # Cross-entropy and masked indexing introduce large transient buffers on top
        # of the full decoder output, especially for vocab-sized logits.
        loss_workspace_bytes = max(
            masked_logits_bytes, batch_size * context_length * act_bytes
        )
        return logits_bytes + loss_workspace_bytes, logits_bytes, loss_workspace_bytes

    if pipeline == "default":
        decoder_bytes = 2 * batch_size * context_length * act_bytes
        return decoder_bytes, 0, 0

    return 0, 0, 0


def estimate_activation_memory_bytes(
    config: ScalingConfig,
) -> tuple[int, int, int, int]:
    encoder_bytes, pair_bytes = estimate_encoder_activation_memory_bytes(config)
    decoder_bytes, logits_bytes, loss_workspace_bytes = estimate_decoder_memory_bytes(
        config
    )
    return encoder_bytes + decoder_bytes, pair_bytes, logits_bytes, loss_workspace_bytes


def estimate_runtime_overhead_bytes(base_memory_bytes: int, training: bool) -> int:
    if training:
        return max(
            MIN_TRAIN_RUNTIME_OVERHEAD_BYTES,
            int(base_memory_bytes * TRAIN_RUNTIME_OVERHEAD_PCT),
        )
    return max(
        MIN_INFERENCE_RUNTIME_OVERHEAD_BYTES,
        int(base_memory_bytes * INFERENCE_RUNTIME_OVERHEAD_PCT),
    )


def estimate_validation_events_from_steps(n_steps: int, val_check_interval) -> float:
    if isinstance(val_check_interval, int):
        if val_check_interval <= 0:
            return 0.0
        return float(n_steps // val_check_interval)
    return math.nan


def estimate_forward_flops(config: ScalingConfig):
    try:
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

            flops = measure_flops(model, model_fwd)
        del model
        gc.collect()
        return flops
    except Exception:
        return math.nan


def build_report_row(config_path: str) -> dict:
    config = ScalingConfig(config_path)
    with torch.device("meta"):
        model = load_model_from_config(config)
        total_params, trainable_params, non_trainable_params = count_parameters(model)

    n_layers, n_heads = get_model_depth_stats(config)
    activation_bytes, pair_bytes, logits_bytes, loss_workspace_bytes = (
        estimate_activation_memory_bytes(config)
    )
    param_bytes = total_params * PARAM_DTYPE_BYTES
    gradient_bytes = trainable_params * PARAM_DTYPE_BYTES
    optimizer_state_bytes = trainable_params * ADAM_STATE_BYTES_PER_PARAM
    encoder_activation_bytes = activation_bytes - logits_bytes - loss_workspace_bytes
    inference_base_bytes = param_bytes + activation_bytes
    train_base_bytes = (
        param_bytes + gradient_bytes + optimizer_state_bytes + activation_bytes
    )
    inference_runtime_overhead_bytes = estimate_runtime_overhead_bytes(
        inference_base_bytes, training=False
    )
    train_runtime_overhead_bytes = estimate_runtime_overhead_bytes(
        train_base_bytes, training=True
    )
    inference_vram_bytes = inference_base_bytes + inference_runtime_overhead_bytes
    train_vram_bytes = train_base_bytes + train_runtime_overhead_bytes
    checkpoint_bytes = param_bytes + optimizer_state_bytes
    effective_batch_size = config.datamodule.batch_size * config.trainer.accumulate_grad
    forward_flops = estimate_forward_flops(config)

    row = {
        "config_path": config_path,
        "title": getattr(config.metadata, "title", config.metadata.run_name),
        "run_name": config.metadata.run_name,
        "pipeline": config.metadata.pipeline.value,
        "model_class": config.model.model_class.value,
        "precision": getattr(config.trainer, "precision", Precision.FULL).value,
        "batch_size": config.datamodule.batch_size,
        "accumulate_grad": config.trainer.accumulate_grad,
        "effective_batch_size": effective_batch_size,
        "context_length": config.model.context_length,
        "total_genes": config.model.total_genes,
        "d_model": config.model.d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": config.model.dropout,
        "mask_pct": config.datamodule.mask_pct,
        "planned_train_steps": config.trainer.n_steps,
        "planned_epochs": config.trainer.n_epochs,
        "val_check_interval": config.trainer.val_check_interval,
        "estimated_validation_events": estimate_validation_events_from_steps(
            config.trainer.n_steps, config.trainer.val_check_interval
        ),
        "checkpoint_enabled": config.trainer.checkpoint,
        "checkpoint_steps": getattr(config.trainer, "checkpoint_steps", math.nan),
        "estimated_checkpoints": (
            math.floor(config.trainer.n_steps / config.trainer.checkpoint_steps)
            if config.trainer.checkpoint
            and getattr(config.trainer, "checkpoint_steps", 0)
            else 0
        ),
        "microbatch_tokens": config.datamodule.batch_size * config.model.context_length,
        "optimizer_step_tokens": effective_batch_size * config.model.context_length,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_gib": bytes_to_gib(param_bytes),
        "gradient_memory_gib": bytes_to_gib(gradient_bytes),
        "optimizer_state_memory_gib": bytes_to_gib(optimizer_state_bytes),
        "encoder_activation_memory_gib": bytes_to_gib(encoder_activation_bytes),
        "decoder_logit_memory_gib": bytes_to_gib(logits_bytes),
        "loss_workspace_memory_gib": bytes_to_gib(loss_workspace_bytes),
        "activation_memory_gib": bytes_to_gib(activation_bytes),
        "bioformer_pair_memory_gib": bytes_to_gib(pair_bytes),
        "inference_runtime_overhead_gib": bytes_to_gib(
            inference_runtime_overhead_bytes
        ),
        "train_runtime_overhead_gib": bytes_to_gib(train_runtime_overhead_bytes),
        "estimated_inference_vram_gib": bytes_to_gib(inference_vram_bytes),
        "estimated_train_vram_gib": bytes_to_gib(train_vram_bytes),
        "estimated_checkpoint_size_gib": bytes_to_gib(checkpoint_bytes),
        "forward_flops": forward_flops,
        "estimated_training_flops_per_step": (
            forward_flops * TRAINING_FLOPS_MULTIPLIER * effective_batch_size
            if not math.isnan(forward_flops)
            else math.nan
        ),
        "estimated_training_flops": (
            forward_flops
            * TRAINING_FLOPS_MULTIPLIER
            * effective_batch_size
            * config.trainer.n_steps
            if not math.isnan(forward_flops)
            else math.nan
        ),
    }

    sweep_metadata = getattr(config, "sweep_metadata", None)
    if sweep_metadata is not None:
        row.update(
            {
                "sweep_mode": sweep_metadata.mode,
                "target_non_embedding_params": sweep_metadata.target_non_embedding_params,
                "actual_non_embedding_params": sweep_metadata.actual_non_embedding_params,
                "target_depth": sweep_metadata.target_depth,
                "target_training_flops": sweep_metadata.target_training_flops,
                "sweep_forward_flops_per_example": sweep_metadata.forward_flops_per_example,
                "sweep_training_flops_per_step": sweep_metadata.training_flops_per_step,
            }
        )

    del model
    gc.collect()
    return row


def collect_config_paths(config_dir: str) -> list[str]:
    return sorted(
        str(Path(config_dir) / filename)
        for filename in os.listdir(config_dir)
        if filename.endswith(".yml")
    )


def generate_report(
    output_csv: str,
    config_file: str | None = None,
    master_folder: str | None = None,
) -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory(prefix="report_configs_") as tmpdir:
        if master_folder:
            generate_grid_configs_master(
                master_folder=master_folder, output_folder=tmpdir
            )
        elif config_file is not None:
            generate_grid_configs(config_file=config_file, output_folder=tmpdir)
        else:
            raise ValueError("Either config_file or master_folder must be provided.")

        config_paths = collect_config_paths(tmpdir)
        for config_path in config_paths:
            rows.append(build_report_row(config_path))

    df = pd.DataFrame(rows)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("--- REPORT ---")
    print(f"Configs analyzed: {len(df)}")
    print(f"Saved CSV: {output_csv}")
    print(
        "Train VRAM estimate range: "
        f"{df['estimated_train_vram_gib'].min():.2f} - {df['estimated_train_vram_gib'].max():.2f} GiB"
    )
    print(
        "Parameter count range: "
        f"{int(df['total_params'].min()):,} - {int(df['total_params'].max()):,}"
    )
    flops_available = df["forward_flops"].notna().sum()
    print(f"Forward FLOPs available for {flops_available}/{len(df)} configs")
    print("--------------")

    return df
