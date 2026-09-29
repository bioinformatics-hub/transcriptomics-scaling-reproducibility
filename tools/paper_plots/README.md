# Paper Plots

This package renders the six main figures and supplementary figures from the
versioned processed tables in `publication/plot_data/`. The default commands
below work without an Aim repository or local `plots/` caches. See
[publication provenance](../../publication/README.md) for source-run mappings
and the snapshot manifest. Shared SVG composition and temporary style helpers
live in `utils.py`; snapshot packaging and input paths live in `data.py`.

The source-export instructions below are for rebuilding analysis tables from
original Aim data. After re-exporting, package and review a new snapshot with
`python -m tools.paper_plots.data --source plots --output /path/to/new_snapshot`
and replace the versioned inputs before rerendering.

Guidelines:

- Keep one Python module per figure, or per tightly related figure family.
- Put shared styling and helpers in `style.py` and `utils.py`.
- Use Arial throughout, keep SVG text editable, and embed TrueType (type 42)
  fonts in PDF and PostScript exports.
- Use a white background with visible left/bottom axis lines and outward-facing
  tick marks. Keep the top/right spines and legend box hidden.
- Label every plotted axis with a quantity and a parenthesized unit, including
  dimensionless quantities (for example, `Learning rate (dimensionless)`).
- Use the shared Wong colour-vision-accessible palette for categorical series.
  Use perceptually uniform sequential maps such as viridis or plasma where a
  continuous scale is required.
- Keep data loading and plot rendering separate when possible so panels are easier to reuse and test.
- Produce SVG only; do not emit a companion PNG.
- Keep complete figures exactly 180 mm wide and no more than 210 mm high,
  except for the 88 mm-wide single-column schematic (Figure 01).
- Keep all text between 5 pt and 7 pt at final figure size.
- Pad every axis beyond its extreme major ticks by 20% of the adjacent inter-tick distance, so no major tick lies on the visual boundary. Apply the same rule in transformed space for logarithmic axes.

## Build all paper figures

From the repository root, run `uv run python -m tools.reproduce_figures`.
This converts the main SVGs to the PDFs used by the manuscript and rebuilds
all numerical supplementary panels. For just one panel family, use the commands
below. Existing paper PDFs are included as reference outputs.

The complete ontology SVG/PDF can be rebuilt with
`uv run python -m tools.paper_plots.supplementary_ontology` (requires Graphviz's
`dot` command). The pruned ontology panel is included under `manuscript/figures/`;
its label-reduction functions are in `tools/ontology.py` and `tools/coarse_labels.py`,
with exact released label mappings beside the manuscript.

## Figures

- Figure 01 is an 88 mm-wide pipeline/model schematic with editable 5–7 pt Arial
  text. Run `uv run python -m tools.paper_plots.figure_01` to rebuild both the
  SVG under `plots/paper/01/` and the embedded-font PDF used by the manuscript.
- Figure 02 compares the Geneformer and scGPT batch-size sweeps in two coordinated panels with shared batch-size and embedding-dimension legends. Run `uv run python -m tools.paper_plots.figure_02` to rebuild its SVG under `plots/paper/`. Alternate cached sweeps can be rendered with the same layout by passing `--geneformer-data`, `--scgpt-data`, and `--output`.
- Figure 03 combines the Geneformer-like and scGPT-like depth/width-ratio scaling plots (03) with their BIOscore trajectories (08). Run `uv run python -m tools.paper_plots.figure_03` to rebuild its SVG under `plots/paper/`.
- Figure 04 compares Geneformer (left) and scGPT MSE (right) learning-rate scaling. Each half shows optimal LR versus model size and depth at the matched last analysed relative compute slice (75% of the target budget), with the LR power-law error series below. Run `uv run python -m tools.paper_plots.figure_04` to rebuild its SVG under `plots/paper/`.
- Figure 05 compares Geneformer and scGPT depth/width scaling. Each half contains training loss by compute with slice guides, the two held-out surface-validation panels, and bootstrap optimum bands. Run `uv run python -m tools.paper_plots.figure_05` to rebuild its SVG under `plots/paper/`.
- Figure 06 shows the Geneformer depth/width response surface in the three projections from plots 10f, 10g, and 10h. Each projection is a single horizontal row: the first two retain every other slice (four panels), while the last retains every other ratio (three panels) and uses its fourth slot for the shared legend. Run `uv run python -m tools.paper_plots.figure_06` to rebuild its SVG under `plots/paper/`.

## Supplementary figures

### Completed ranked context-grid run

The `geneformer_like_missing` run fills context 1,024, width 512, depth 4,
bringing both the training and downstream ranked datasets to 100 architectures.
After downloading its Aim repository to `aim-repo/geneformer_like_missing/`
and its config and downstream CSV to the corresponding architecture directory
under `checkpoints/geneformer_like/`, run
`uv run python -m tools.incorporate_geneformer_missing`.
This merges the two resumed Aim segments by metric/context/step, keeps the latest
observation in their overlap, and assigns one logical run ID. It replaces any
existing data for that architecture, so it can be rerun without duplicating it.
The script checks all 50,000 training steps and rebuilds both prepared datasets.
Repeat this incorporation after exporting the original Aim repository afresh.
Then rebuild `plotting.geneformer_like`, Figure 03, and the supplementary figures.
Other ranked runs still end early, so the common loss-comparison step remains
37,499.

Run `uv run python -m tools.paper_plots.supplementary` to rebuild all paper-styled vector plots used by `manuscript/supplementary.tex`: context-length training curves, downstream-component trajectories, learning-rate diagnostics, and depth/width diagnostics. The supplementary PDF can then be compiled from `manuscript/` with `latexmk -pdf supplementary.tex`.

### Corrected binned depth/width run

The paper uses `aim-repo/scgpt_dw_rerun-repo/`, replacing the earlier
`scgpt_dw` sweep. `plotting.scgpt_dw` now exports to `plots/scgpt_dw_rerun/`;
Figure 05 and the supplementary figures read only those rerun caches.
The 52 Aim segments cover all 49 architectures. The existing DW exporter merges
retry segments by architecture and step, preferring the longest segment in an
overlap. All architectures reach their configured stopping step.

Rebuild from the repository root:

```sh
uv run python -m plotting.scgpt_dw --force
uv run python -m tools.audit_scgpt_dw_rerun
uv run python -m tools.paper_plots.figure_05
rsvg-convert --format=pdf --output manuscript/figures/05_depth_width_scaling.pdf plots/paper/05_depth_width_scaling.svg
uv run python -m tools.paper_plots.supplementary
cd manuscript
latexmk -pdf manuscript.tex
latexmk -pdf supplementary.tex
```

The audit checks every segment's learning rate against the unrounded corrected
size/depth rule, the 500-step warm-up/constant schedule, the complete architecture
grid and training endpoints. It writes source hashes and the numerical surface,
cross-validation, warm-up and bootstrap diagnostics to
`manuscript/scgpt_dw_rerun_audit.json`. The 230 retained binned analysis observations
reflect the existing 0.99 slice-completion filter and compute/step caps, not
missing architectures. Bootstrap intervals describe row-resampling sensitivity;
the central 95% reference interval in the text differs from the 5th–95th
percentile bands in Figure 05.
