"""Verify that the versioned figure inputs match their provenance manifest."""

import json

import polars as pl

from tools.paper_plots.data import DATA_DIR, INPUTS, paper_data, sha256


def test_paper_snapshot_integrity():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    assert {entry["source"] for entry in manifest} == set(INPUTS)
    for entry in manifest:
        path = DATA_DIR / entry["file"]
        assert path == paper_data(entry["source"])
        assert sha256(path) == entry["sha256"]
        table = pl.scan_parquet(path)
        assert table.select(pl.len()).collect().item() == entry["rows"]
        assert table.collect_schema().names() == entry["columns"]
        assert "source_csv" not in entry["columns"]
