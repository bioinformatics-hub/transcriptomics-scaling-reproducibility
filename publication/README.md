# Reproducing the published runs

The release targets Figures 01–06 and the figures used by
`manuscript/supplementary.tex`. Recipes in `recipes/` expand to ordinary training
configs; they contain recorded experiment settings rather than toy runs.

## Generate and run locally

From the repository root, after `uv sync`:

```sh
uv run python -m tools.publication_configs --run scgpt_like --output configs/scgpt_like
uv run python main.py train --config configs/scgpt_like/config_0.yml -A
```

The output directory must not already exist. To run the other configurations,
pass their files individually, or use `--config-dir configs/scgpt_like --id 1`.
Use `train_with_downstream` instead of `train` to also run downstream evaluation.
Prepare statistics once before launching concurrent runs; `-A` overwrites them.

Supply the human experiment from CELLxGENE Census release **2025-11-08** at
`data/census/census_data/homo_sapiens`, or change `paths.path_to_census` in the
recipe. The models expect 61,497 genes in the original order. A smaller cell
subset can exercise the pipeline, but does not reproduce the published data.
`-A` computes the HVG, mean-expression, cell-type and dataset-ID CSVs under
`data/csvs/`; create that directory first. The repository includes the Cell
Ontology used by downstream label aggregation. Training writes checkpoints, Aim logs and coarse-label outputs under
`output/reproduction/`, separate from original local results; run from the repository root so relative paths resolve correctly.
The original `bf16-mixed` precision is retained; change it to `full` for hardware
that does not support the original precision (this changes numerical execution).

## Run inventory and behavior

| Recipe | Configurations | `metadata.training_behavior` | Figure use |
| --- | ---: | --- | --- |
| `geneformer_batch` | 20 | `legacy_padding` | 02 |
| `scgpt_batch` | 20 | `legacy_padding` | 02 |
| `geneformer_like` | 99 | `legacy` | 03, supplement |
| `geneformer_like_missing` | 1 | `corrected` | completes the ranked grid |
| `scgpt_like` | 100 | `legacy` | 03, supplement |
| `geneformer_lr` | 60 | `legacy` | 04, supplement |
| `scgpt_lr` | 60 | `legacy` | 04, supplement |
| `geneformer_dw` | 49 | `legacy` | 05–06, supplement |
| `scgpt_dw_rerun` | 49 | `corrected` | 05, supplement |

The three modes represent separate historical states:

- `legacy`: pre-June 18 losses and pre-August 22 input/attention behavior. Binned
  MSE selects predicted-active masked positions. BCE includes the constant
  contribution from unmasked positions. Binned selection uses independent
  floor rounding, the old padding ID and mask generation; attention is not
  padding-aware. Historical empty-mask loss behavior is retained.
- `legacy_padding`: June-corrected losses, with the older input/attention path.
  Binned MSE selects target-active positions and BCE uses masked positions only.
- `corrected`: current losses, exact context length, padding-excluding masking
  and padding-aware attention. This remains the default for existing configs
  without `training_behavior`.

Ranked-model cross-entropy is unaffected by the June loss change; its older
attention behavior still requires an explicit historical mode. The selector is
used by model construction, the normal training/validation steps and the binned
dataset iterator. It is not necessary to call `_old` methods manually.

## Provenance and limits

These recipes were reconstructed from locally available Aim config records on
2026-09-22. Each entry retains its source run hashes. Common settings are stored
once under `defaults`, with recursive per-run overrides under `runs[].config`.
The expansion tool does not need Aim, a cluster, or the original Git history.

The public implementation labels are `legacy`, `legacy_padding`, and
`corrected`, as defined above. They identify scientific behavior rather than a
repository revision. Each recipe names a stable `behavior_reference_id` linking
it to a synthetic reference case in
`test/reference_data/training_behavior/manifest.json`.

Compatibility modes were checked against the archived implementations used to
deliver the experiments. The reference tests and validation results accompany
this release. The detailed mapping to archived source history is retained in
the private research archive and is not needed to use this repository.

The original context/model-size and learning-rate runs used the earlier
implementation. Batch-size runs used corrected losses with older padding and
attention behavior. The missing ranked architecture and replacement binned
depth/width run used corrected padding and attention behavior as well. In
particular, the replacement binned depth/width run processes 256 positions at
nominal context 256, whereas the earlier binned sweeps process 255.

These assignments were established by inspecting archived config-delivery
implementations, with the maintainer confirming that configurations and code
were delivered together. The experiment records did not independently capture
the executing source revision, so this evidence cannot exclude unrecorded
cluster edits. This provenance limitation remains even though public
reproduction does not require access to the archive.

### Standalone compatibility checks

`behavior_validation.json` records the completed historical comparisons: exact
loss and parameter-gradient agreement for nine source snapshots, plus exact
binned-batch agreement for four of them. All compared losses were finite.
The original commit-dependent audit remains in the private research archive.

The public reference set, `publication-training-semantics-v1`, contains saved
synthetic inputs, weights, encoder outputs, losses, gradients and binned batches
captured from those archived implementations. Run the independent checks with:

```sh
pytest test/test_training_references.py test/test_publication_configs.py
```

No Git executable, repository history, Census data or cluster access is needed.
The reference manifest records capture-library versions, seeds, file checksums
and floating-point tolerances. The tests load saved weights rather than depend
on reproducing a particular library's weight initialization. Dataset outputs
are checked exactly; floating-point model results allow small numerical
variation across environments.

Matching saved reference outputs is a standalone regression check, distinct
from the historical source-to-source comparison. Neither establishes a full
optimizer trajectory, resume history, historical dependency environment or
end-to-end downstream reproduction. Full training and downstream evaluation have not been rerun for this release.

### Portable identifiers and checksums

`recipe_manifest.json` records recipe file SHA-256 checksums, configuration
counts, behavior modes and reference IDs. `plot_data/manifest.json` records the
processed data checksums and table schemas. These identify file contents and
remain valid when the files move to a new repository.

`source_run_hashes` in recipes and run IDs in plotting data are **experiment
identifiers, not Git commits**. They are retained to connect measured results
and retry segments to their experiment configurations. `source_repository`
names the original Aim data source, not a required public Git repository.

Portability changes are limited to relative input/output paths, removing stored
user/host names, resetting `resume_training` to false and allocation `max_time`
to null, adding the dataset-ID CSV path needed by current preprocessing, and
normalizing retry titles to the publication run family. Configs retain the
original widths, depths, learning rates, schedules, steps and compute metadata.
Two 500-step ranked pilot configs are omitted. Identical retry configurations
are consolidated; their hashes are preserved. The September missing architecture
is kept separately because it belongs to a different behavior period.

These recipes specify planned executions, not the precise interruption history
of the original jobs. Some published trajectories ended early, and the original
runs did not record a universal random seed. Rerunning these settings therefore
does not promise bitwise-identical results or identical stopping points.

## Figures

See [the figure instructions](../tools/paper_plots/README.md) for the six main
figures, supplementary figures and preparation of the original analysis caches.
Paper renderers default to the versioned `plot_data/` snapshot, which includes
all observed rows from the twelve required processed tables. CSV inputs were
converted to Parquet with lossless Zstandard compression. `manifest.json` records
source and packaged hashes, row counts, columns and any removed host/path fields.
The packager verifies equality after writing every table.

To rebuild a snapshot from freshly prepared caches, use:

```sh
python -m tools.paper_plots.data --source plots --output /path/to/new_snapshot
```

Review the generated manifest before replacing the versioned snapshot. Training
configs do not substitute for these measured results. Large Census, Aim and checkpoint
data are not included in Git.
