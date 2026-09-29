# Transcriptomics scaling: paper reproduction

Code, portable experiment configurations and processed data for **Scaling recipes
for single-cell RNA sequencing foundation models: when do scaling laws hold?**

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
```

The checked-in lockfile preserves the release environment. It selects CPU PyTorch
on macOS and CUDA 12.8 on Linux. Training requires the Census data and suitable
compute; plotting the included results does not require training or checkpoints.

## Rebuild the figures

Install Arial and the `rsvg-convert` command (provided by librsvg), then run:

```sh
uv run python -m tools.reproduce_figures
```

This rebuilds all six main figures and the numerical supplementary figures from
`publication/plot_data/`, writing the PDFs used by the manuscript to
`manuscript/figures/` and intermediate SVGs to `plots/`. The ontology figure assets
are included; their separate reproduction instructions and per-figure commands
are in [the figure guide](tools/paper_plots/README.md).

With a TeX installation including `latexmk`, build the paper and supplement:

```sh
latexmk -cd -pdf manuscript/manuscript.tex
latexmk -cd -pdf manuscript/supplementary.tex
```

## Reproduce training and evaluation

The [experiment guide](publication/README.md) documents all 458 configurations,
historical loss/masking modes, data requirements and reproducibility limits.
Supply the recorded Census release and gene order before running:

```sh
uv run python -m tools.publication_configs --run scgpt_like --output configs/scgpt_like
mkdir -p data/csvs
uv run python main.py train_with_downstream --config configs/scgpt_like/config_0.yml -A
```

`-A` creates shared preprocessing statistics and should run only once before
concurrent jobs. Use `train` for training alone, `downstream_tasks` to consume saved
embeddings, and `--resume` to resume a training run. `python main.py --help` lists
supported commands. [Optional Slurm execution](docs/cluster_workflows.md) uses a
single generic launcher with your allocation's resource settings.

## Validate the release

```sh
uv run python -m pytest test -q
```

The included checks use synthetic fixtures and processed data; they do not download
Census or launch the paper experiments (resume tests use tiny synthetic training runs). They check historical behavior, configuration expansion,
plot-data checksums, masking/losses, numerical analysis and plotting contracts.
Full training and downstream evaluation have not been rerun for this release.

## Files

- `core/`, `run/`, `main.py`: training and downstream evaluation.
- `publication/`: experiment recipes, measured plotting data and provenance.
- `plotting/`, `tools/`: preprocessing, numerical analyses and paper figures.
- `test/`: local regression checks and historical-behavior reference fixtures.
- `manuscript/`: paper sources, figure assets and supplementary data files.
- `cl.owl`: the Cell Ontology snapshot used for label aggregation.

Raw Census matrices, original Aim repositories and model checkpoints are separate
from this repository. Experiment IDs in the data and recipes identify runs, not
Git revisions. This repository has independent history.

## License

Original project code and processed plotting data are provided under the
[MIT license](LICENSE). Cell Ontology and the Springer manuscript template retain
their existing third-party licenses and notices. Arial is not redistributed.
