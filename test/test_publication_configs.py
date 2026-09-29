from pathlib import Path
import hashlib
import json
import warnings

import pytest
import yaml

from core.config import ScalingConfig
from tools.publication_configs import RECIPE_DIR, expand_recipe

EXPECTED_COUNTS = {
    "geneformer_like": 99,
    "geneformer_like_missing": 1,
    "scgpt_like": 100,
    "geneformer_lr": 60,
    "scgpt_lr": 60,
    "geneformer_batch": 20,
    "scgpt_batch": 20,
    "geneformer_dw": 49,
    "scgpt_dw_rerun": 49,
}


@pytest.mark.parametrize("family,count", EXPECTED_COUNTS.items())
def test_portable_recipes_are_complete_and_loadable(family, count):
    recipe = yaml.safe_load((RECIPE_DIR / f"{family}.yml").read_text())
    configs = expand_recipe(recipe)
    assert len(configs) == count
    assert len({c["metadata"]["run_name"] for c in configs}) == count
    for config in configs:
        assert not config["trainer"].get("resume_training", False)
        assert config["trainer"].get("max_time") is None
        for path in config["paths"].values():
            assert path is None or not Path(path).is_absolute()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = ScalingConfig(config)
        assert loaded.metadata.training_behavior.value in {
            "legacy",
            "legacy_padding",
            "corrected",
        }


def test_recipe_checksums_and_behavior_references():
    root = RECIPE_DIR.parents[1]
    manifest = json.loads((RECIPE_DIR.parent / "recipe_manifest.json").read_text())
    references_path = root / "test/reference_data/training_behavior/manifest.json"
    references = json.loads(references_path.read_text())
    cases = {case["id"]: case for case in references["cases"]}
    report = json.loads((RECIPE_DIR.parent / "behavior_validation.json").read_text())
    assert (
        report["reference_manifest_sha256"]
        == hashlib.sha256(references_path.read_bytes()).hexdigest()
    )
    assert {entry["file"] for entry in manifest["recipes"]} == {
        str(path.relative_to(RECIPE_DIR.parent)) for path in RECIPE_DIR.glob("*.yml")
    }
    for entry in manifest["recipes"]:
        path = RECIPE_DIR.parent / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        recipe = yaml.safe_load(path.read_text())
        assert len(recipe["runs"]) == entry["config_count"]
        assert recipe["behavior_reference_id"] == entry["behavior_reference_id"]
        mode = recipe["defaults"]["metadata"]["training_behavior"]
        assert mode == entry["training_behavior"]
        reference_metadata = cases[recipe["behavior_reference_id"]]["config"][
            "metadata"
        ]
        assert mode == reference_metadata["training_behavior"]
        assert (
            recipe["defaults"]["metadata"]["pipeline"] == reference_metadata["pipeline"]
        )
