"""Expand portable publication recipes into normal training YAML files."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

RECIPE_DIR = Path(__file__).resolve().parents[1] / "publication" / "recipes"


def merge_config(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_recipe(recipe: dict) -> list[dict]:
    return [merge_config(recipe["defaults"], run["config"]) for run in recipe["runs"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=[p.stem for p in sorted(RECIPE_DIR.glob("*.yml"))],
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are refused.",
    )
    args = parser.parse_args()
    recipe = yaml.safe_load((RECIPE_DIR / f"{args.run}.yml").read_text())
    configs = expand_recipe(recipe)
    args.output.mkdir(parents=True, exist_ok=False)
    for index, config in enumerate(configs):
        (args.output / f"config_{index}.yml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
    print(f"Generated {len(configs)} configs in {args.output}")


if __name__ == "__main__":
    main()
