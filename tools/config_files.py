from pathlib import Path


def _config_sort_key(path: Path) -> tuple[str, int | str]:
    stem = path.stem
    prefix, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit():
        return prefix, int(suffix)
    return stem, stem


def list_config_files(config_dir: str | Path) -> list[Path]:
    path = Path(config_dir)
    return sorted(
        (item for item in path.iterdir() if item.suffix == ".yml"),
        key=_config_sort_key,
    )
