import numpy as np

from core.config import MaskStrategy
from core.datamodule import BinnedMaskedDataset


def masking_dataset(strategy: MaskStrategy, mask_pct: float = 1.0):
    dataset = object.__new__(BinnedMaskedDataset)
    dataset.mask_strategy = strategy
    dataset.mask_pct = mask_pct
    dataset.total_genes = 3
    return dataset


def test_uniform_mask_never_selects_padding():
    dataset = masking_dataset(MaskStrategy.UNIFORM)
    expressions = np.array([[2, 1, 0, 0]])
    gene_ids = np.array([[0, 1, 2, dataset.total_genes]])

    mask = dataset.mask(expressions, gene_ids)

    np.testing.assert_array_equal(mask, [[True, True, True, False]])


def test_equal_prop_mask_handles_missing_zero_group_and_excludes_padding():
    dataset = masking_dataset(MaskStrategy.EQUAL_PROP)
    expressions = np.array([[3, 2, 1, 0]])
    gene_ids = np.array([[0, 1, 2, dataset.total_genes]])

    mask = dataset.mask(expressions, gene_ids)

    np.testing.assert_array_equal(mask, [[True, True, True, False]])


def test_equal_prop_mask_handles_missing_active_group():
    dataset = masking_dataset(MaskStrategy.EQUAL_PROP)
    expressions = np.zeros((1, 3), dtype=int)
    gene_ids = np.array([[0, 1, dataset.total_genes]])

    mask = dataset.mask(expressions, gene_ids)

    np.testing.assert_array_equal(mask, [[True, True, False]])


def test_active_selection_sizes_preserve_context_length():
    dataset = masking_dataset(MaskStrategy.UNIFORM, mask_pct=0.15)
    dataset.context_length = 2048

    active, inactive = dataset.selection_sizes()
    old_active, old_inactive = dataset.selection_sizes_old()

    assert active + inactive == dataset.context_length
    assert old_active + old_inactive == 2047
