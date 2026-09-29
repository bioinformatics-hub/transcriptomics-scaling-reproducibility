"""
Print CELLxGENE Census release metadata from a local SOMA copy.

Examples:
    python tools/census_release_info.py /path/to/soma/census_data/homo_sapiens
    python tools/census_release_info.py --summary-uri /path/to/soma/census_info/summary
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import tiledbsoma as soma


INFO_KEYS = ("release", "version", "date", "build")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read census_info/summary for a CELLxGENE Census SOMA release."
    )
    parser.add_argument(
        "census_uri",
        nargs="?",
        help="Path/URI to soma/census_data/homo_sapiens.",
    )
    parser.add_argument(
        "--summary-uri",
        help="Path/URI to soma/census_info/summary. Overrides census_uri.",
    )
    parser.add_argument(
        "--tiledb-config-pickle",
        type=Path,
        help="Optional pickle containing a TileDB config dict.",
    )
    return parser.parse_args()


def infer_summary_uri(census_uri: str) -> str:
    stripped = census_uri.rstrip("/")
    suffix = "/census_data/homo_sapiens"
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)] + "/census_info/summary"
    return str(Path(stripped).parent.parent / "census_info" / "summary")


def make_context(tiledb_config_pickle: Path | None) -> soma.SOMATileDBContext | None:
    if tiledb_config_pickle is None:
        return None
    with tiledb_config_pickle.open("rb") as handle:
        tiledb_config = pickle.load(handle)
    return soma.SOMATileDBContext().replace(tiledb_config=tiledb_config)


def read_summary(summary_uri: str, context: soma.SOMATileDBContext | None):
    with soma.open(summary_uri, context=context) as summary:
        return summary.read().concat().to_pandas()


def matching_metadata_rows(summary):
    text = summary.astype(str)
    mask = text.apply(
        lambda column: column.str.contains("|".join(INFO_KEYS), case=False, na=False)
    ).any(axis=1)
    return summary.loc[mask]


def main() -> None:
    args = parse_args()
    if args.summary_uri:
        summary_uri = args.summary_uri
    elif args.census_uri:
        summary_uri = infer_summary_uri(args.census_uri)
    else:
        raise SystemExit("Provide census_uri or --summary-uri.")

    context = make_context(args.tiledb_config_pickle)
    summary = read_summary(summary_uri, context)
    matches = matching_metadata_rows(summary)

    print(f"Summary URI: {summary_uri}\n")
    print("Full census_info/summary:")
    print(summary.to_string(index=False))

    print("\nRows mentioning release/version/date/build:")
    if matches.empty:
        print("No matching rows found.")
    else:
        print(matches.to_string(index=False))


if __name__ == "__main__":
    main()
