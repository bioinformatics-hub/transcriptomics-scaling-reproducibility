from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from aim import Repo

from plotting.utils import ensure_parent_dir, flatten_dict


def _context_to_dict(context: object) -> dict[str, object]:
    if context is None:
        return {}
    if hasattr(context, "to_dict"):
        context_dict = context.to_dict()
        if isinstance(context_dict, dict):
            return context_dict
    try:
        return dict(context)
    except (TypeError, ValueError):
        return {}


def export_metrics_csv(
    aim_repo_path: Path,
    output_csv: Path,
    metrics_to_extract: set[str],
    run_query: str | None = None,
    config_keys: set[str] | None = None,
    run_fields: set[str] | None = None,
) -> None:
    repo = Repo(str(aim_repo_path))
    rows: list[dict[str, object]] = []
    runs = repo.query_runs(run_query).iter_runs() if run_query else repo.iter_runs()
    extra_fields_by_hash: dict[str, dict[str, object]] = {}

    if run_query and run_fields:
        query_df = repo.query_runs(run_query).dataframe()
        if query_df is not None:
            selected_columns = ["hash", *sorted(run_fields)]
            available_columns = [column for column in selected_columns if column in query_df.columns]
            for record in query_df[available_columns].to_dict(orient="records"):
                run_hash = record.pop("hash")
                extra_fields_by_hash[str(run_hash)] = record

    for run_entry in runs:
        run = run_entry.run if hasattr(run_entry, "run") else run_entry
        config = run.get("config", default={}) or {}
        flat_config = flatten_dict(config) if isinstance(config, dict) else {}
        if config_keys is not None:
            flat_config = {
                key: value
                for key, value in flat_config.items()
                if key in config_keys
            }
        run_common = {
            "run_hash": run.hash,
            "run_name": run.name,
            "experiment": run.experiment if run.experiment else "default",
            **flat_config,
            **extra_fields_by_hash.get(str(run.hash), {}),
        }

        for metric_sequence in run.metrics():
            metric_name = metric_sequence.name
            if metric_name not in metrics_to_extract:
                continue

            metric_df = metric_sequence.dataframe()
            if metric_df.empty:
                continue
            context_dict = _context_to_dict(getattr(metric_sequence, "context", None))
            flat_context = flatten_dict(context_dict, prefix="context")
            for _, row in metric_df.iterrows():
                rows.append(
                    {
                        **run_common,
                        "metric_name": metric_name,
                        **flat_context,
                        "step": row["step"],
                        "epoch": row.get("epoch"),
                        "value": row["value"],
                        "timestamp": row.get("timestamp", row.get("time")),
                    }
                )

    if not rows:
        rows = _export_metrics_rows_from_chunks(
            repo,
            metrics_to_extract=metrics_to_extract,
            run_query=run_query,
            config_keys=config_keys,
            run_fields=run_fields,
        )

    if not rows:
        repo.close()
        raise ValueError(f"No matching metrics found in Aim repo: {aim_repo_path}")

    ensure_parent_dir(output_csv)
    pl.DataFrame(rows).write_csv(output_csv)
    repo.close()


def _run_query_metadata_title(run_query: str | None) -> str | None:
    if run_query is None:
        return None
    match = re.fullmatch(r"\s*run\.config\.metadata\.title\s*==\s*['\"]([^'\"]+)['\"]\s*", run_query)
    if match is None:
        raise ValueError(
            "Chunk-backed Aim export only supports run.config.metadata.title equality queries. "
            f"Received: {run_query}"
        )
    return match.group(1)


def _chunk_run_tree(repo: Repo, run_hash: str):
    return (
        repo.request_tree("meta", run_hash, read_only=True)
        .subtree("meta")
        .subtree("chunks")
        .subtree(run_hash)
    )


def _collect_chunk_attrs(run_tree) -> tuple[dict[str, object], dict[str, object]]:
    attrs = run_tree.subtree("attrs")
    config = attrs.collect("config")
    hparams = attrs.collect("hparams")
    if not isinstance(config, dict):
        config = {}
    if not isinstance(hparams, dict):
        hparams = {}
    return config, hparams


def _train_context_indices(run_tree) -> list[int]:
    contexts = run_tree.get("contexts", {})
    train_contexts = [
        int(context_idx)
        for context_idx, context in contexts.items()
        if isinstance(context, dict) and context.get("subset") == "train"
    ]
    if train_contexts:
        return train_contexts
    return [int(context_idx) for context_idx, context in contexts.items() if context in ({}, None)]


def _sequence_pairs_from_chunk(repo: Repo, run_hash: str, context_idx: int, metric_name: str) -> list[tuple[int, float, object]]:
    sequence_tree = (
        repo.request_tree("seqs", run_hash, read_only=True)
        .subtree("seqs")
        .subtree(("v2", "chunks", run_hash, context_idx, metric_name))
    )
    steps = dict(sequence_tree.array("step").items())
    values = dict(sequence_tree.array("val").items())
    try:
        timestamps = dict(sequence_tree.array("time").items())
    except (KeyError, IndexError):
        timestamps = {}
    return sorted(
        (int(step), float(values[step_hash]), timestamps.get(step_hash))
        for step_hash, step in steps.items()
        if step_hash in values
    )


def _export_metrics_rows_from_chunks(
    repo: Repo,
    *,
    metrics_to_extract: set[str],
    run_query: str | None,
    config_keys: set[str] | None,
    run_fields: set[str] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata_title = _run_query_metadata_title(run_query)

    for run_hash in sorted(repo.list_all_runs()):
        run_tree = _chunk_run_tree(repo, run_hash)
        config, hparams = _collect_chunk_attrs(run_tree)
        flat_config = flatten_dict(config)
        if metadata_title is not None and flat_config.get("metadata.title") != metadata_title:
            continue

        selected_config = flat_config if config_keys is None else {
            key: value for key, value in flat_config.items() if key in config_keys
        }
        flat_hparams = flatten_dict(hparams, prefix="hparams")
        selected_hparams = flat_hparams if run_fields is None else {
            key: value for key, value in flat_hparams.items() if key in run_fields
        }
        run_common = {
            "run_hash": run_hash,
            "run_name": flat_config.get("metadata.run_name", f"Run: {run_hash}"),
            "experiment": flat_config.get("metadata.title", "default"),
            **selected_config,
            **selected_hparams,
        }
        traces = run_tree.get("traces", {})

        for context_idx in _train_context_indices(run_tree):
            context_traces = traces.get(context_idx, {})
            if not isinstance(context_traces, dict):
                continue

            for metric_name in sorted(metrics_to_extract):
                if metric_name not in context_traces:
                    continue
                sequence_pairs = _sequence_pairs_from_chunk(repo, run_hash, context_idx, metric_name)
                rows.extend(
                    {
                        **run_common,
                        "metric_name": metric_name,
                        "context.subset": "train",
                        "step": step,
                        "epoch": None,
                        "value": value,
                        "timestamp": timestamp,
                    }
                    for step, value, timestamp in sequence_pairs
                )

    return rows
