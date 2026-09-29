"""Render the complete observed Cell Ontology hierarchy as vector artwork.

The graph contains every Cell Ontology term in the released fine-to-coarse
mapping and every explicit CL-to-CL ``rdfs:subClassOf`` ancestor needed to
connect those terms to the ontology root.  The special ``unknown`` mapping is
reported in the graph note but is not represented as an ontology node.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import graphviz

from tools.ontology import (
    all_ancestors,
    build_parents_from_cl_owl,
    propagate_leaf_counts,
)


DEFAULT_MAPPING = Path("manuscript/cell_type_to_coarse_k30.csv")
DEFAULT_COUNTS = Path("manuscript/validation_cell_type_counts.csv")
DEFAULT_ONTOLOGY = Path("cl.owl")
DEFAULT_OUTPUT_STEM = Path("manuscript/complete_observed_cell_ontology_k30")


def _safe_id(term_id: str) -> str:
    return term_id.replace(":", "_")


def _read_mapping(path: Path) -> tuple[list[dict[str, str]], set[str], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {"leaf_id", "leaf_name", "coarse_id", "coarse_name"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(
            f"{path} must contain exactly these columns: {sorted(expected)}"
        )

    fine_ids = {row["leaf_id"] for row in rows if row["leaf_id"].startswith("CL:")}
    coarse_ids = {
        row["coarse_id"] for row in rows if row["coarse_id"].startswith("CL:")
    }
    return rows, fine_ids, coarse_ids


def _read_counts(path: Path) -> Counter[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {"leaf_id", "n_validation_cells"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(
            f"{path} must contain exactly these columns: {sorted(expected)}"
        )

    counts = Counter({row["leaf_id"]: int(row["n_validation_cells"]) for row in rows})
    if any(count < 0 for count in counts.values()):
        raise ValueError(f"{path} contains a negative cell count")
    return counts


def build_graph(
    mapping_path: Path, counts_path: Path, ontology_path: Path
) -> graphviz.Digraph:
    rows, fine_ids, coarse_ids = _read_mapping(mapping_path)
    leaf_counts = _read_counts(counts_path)
    parents, labels = build_parents_from_cl_owl(str(ontology_path))

    mapping_ids = {row["leaf_id"] for row in rows}
    if mapping_ids != set(leaf_counts):
        missing_counts = sorted(mapping_ids - set(leaf_counts))
        extra_counts = sorted(set(leaf_counts) - mapping_ids)
        raise ValueError(
            f"Counts do not match mapping; missing={missing_counts}, extra={extra_counts}"
        )
    if sum(leaf_counts.values()) != 30_000:
        raise ValueError(
            f"Expected 30,000 validation cells, found {sum(leaf_counts.values()):,}"
        )

    propagated_counts = propagate_leaf_counts(leaf_counts, parents)

    visible_nodes = set(fine_ids)
    ancestor_memo: dict[str, set[str]] = {}
    for term_id in sorted(fine_ids):
        visible_nodes.update(all_ancestors(term_id, parents, ancestor_memo))

    missing = sorted((fine_ids | coarse_ids) - labels.keys())
    if missing:
        raise ValueError(f"Terms missing from {ontology_path}: {', '.join(missing)}")

    invalid_mappings = []
    for row in rows:
        leaf_id = row["leaf_id"]
        coarse_id = row["coarse_id"]
        if not leaf_id.startswith("CL:") or not coarse_id.startswith("CL:"):
            continue
        if coarse_id != leaf_id and coarse_id not in all_ancestors(
            leaf_id, parents, ancestor_memo
        ):
            invalid_mappings.append(f"{leaf_id} -> {coarse_id}")
    if invalid_mappings:
        raise ValueError(
            "Coarse terms must be ancestors of their fine terms: "
            + ", ".join(invalid_mappings)
        )

    edges = {
        (parent, child)
        for child in visible_nodes
        for parent in parents.get(child, set())
        if parent in visible_nodes
    }
    dot = graphviz.Digraph(
        name="complete_observed_cell_ontology_k30",
        comment=(
            "Complete observed Cell Ontology hierarchy used for the k=30 "
            "frequency-adaptive coarse-label reduction"
        ),
        format="svg",
        strict=True,
    )
    dot.attr(
        rankdir="TB",
        nodesep="0.2",
        ranksep="0.5",
        concentrate="true",
        fontsize="24",
        overlap="false",
        splines="polyline",
    )

    for term_id in sorted(visible_nodes):
        is_observed = term_id in fine_ids
        is_anchor = term_id in coarse_ids
        category = (
            "retained coarse anchor"
            if is_anchor
            else "observed fine term"
            if is_observed
            else "ancestor included for context"
        )
        attributes = {
            "tooltip": f"{labels.get(term_id, term_id)} ({term_id}); {category}",
            "URL": f"https://purl.obolibrary.org/obo/{term_id.replace(':', '_')}",
            "target": "_blank",
        }
        if is_observed:
            attributes.update(
                style="filled",
                fillcolor="#AED6F1",
                shape="box",
                penwidth="1.5",
            )
        else:
            attributes.update(
                shape="ellipse",
                color="#808080",
                fontcolor="#444444",
                fontsize="10",
            )

        dot.node(
            _safe_id(term_id),
            (
                f"{labels.get(term_id, term_id)}\n({term_id})\n"
                f"Cells: {propagated_counts.get(term_id, 0):,}"
            ),
            **attributes,
        )

    for parent, child in sorted(edges):
        dot.edge(_safe_id(parent), _safe_id(child), color="#AAAAAA")

    return dot


def render(
    mapping_path: Path, counts_path: Path, ontology_path: Path, output_stem: Path
) -> None:
    dot = build_graph(mapping_path, counts_path, ontology_path)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    for output_format in ("svg", "pdf"):
        output_path = output_stem.with_suffix(f".{output_format}")
        output_path.write_bytes(dot.pipe(format=output_format))
        print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    render(args.mapping, args.counts, args.ontology, args.output_stem)


if __name__ == "__main__":
    main()
