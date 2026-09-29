from __future__ import annotations

from contextlib import contextmanager
import xml.etree.ElementTree as ET


@contextmanager
def temporary_plot_settings(module, values: dict[str, object]):
    """Restore plot module settings even when rendering fails."""
    original = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for element in root.iter():
        old_id = element.get("id")
        if old_id:
            new_id = f"{prefix}_{old_id}"
            id_map[old_id] = new_id
            element.set("id", new_id)
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            updated = value
            for old_id, new_id in id_map.items():
                updated = updated.replace(f"url(#{old_id})", f"url(#{new_id})")
                if updated == f"#{old_id}":
                    updated = f"#{new_id}"
            if updated != value:
                element.set(key, updated)
