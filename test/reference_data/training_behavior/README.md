# Historical synthetic reference set

`publication-training-semantics-v1` contains nine small CPU cases captured from
the archived implementations used to deliver the corresponding experiment
families. The reference values were calculated by those implementations, not by
the release implementation under test. All data here are synthetic.

Each NPZ file is a collection of numerical arrays, readable with NumPy and
`allow_pickle=False`:

- `state/*`: model parameters and buffers before the forward pass.
- `input/*`: gene IDs, expression values and prediction masks.
- `expected/encoder`: unmasked-input encoder output, including padded positions.
- `loss/*`: training-step loss components.
- `gradient/*`: parameter gradients from the sum of loss components. Absent
  gradients must remain absent; the tests check the complete gradient-name set.
- `dataset/input` and `dataset/output/*`: original expression counts and the
  binned iterator's expression bins, gene IDs, prediction masks and dataset IDs.
  These are present only for binned cases. Labels are disabled and the synthetic
  ontology label is `unknown`.

`manifest.json` contains the case configs, stable reference IDs, SHA-256 hashes,
capture versions and seeds. Model comparisons use relative tolerance 1e-5 and
absolute tolerance 1e-6 for floating-point portability; discrete dataset outputs
must match exactly. The capture-time source-to-source audit matched exactly and
is summarized in `publication/behavior_validation.json`.

Run from the repository root:

```sh
pytest test/test_training_references.py
```

These tests do not open Git history, fetch data or run a training/downstream
workflow. They cover small transformer examples with padding, zero dropout and
uniform binned masking; they do not cover every model, batch, optimizer or
runtime configuration. Do not regenerate expected values from the code under
test. Any deliberate change to reference semantics needs a new documented
reference set and validation against the independently archived implementation.

The detailed source-history mapping and capture scripts are preserved in the
private research archive. They are not required to execute the public tests.
