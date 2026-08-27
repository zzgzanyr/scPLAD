#!/usr/bin/env python3
"""Export the four Figure 1 schematic panels as independent labeled SVGs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image


ARCHIVE = Path(__file__).resolve().parents[2]
SOURCE = ARCHIVE / "final_figures" / "Fig1" / "figure1_final.svg"
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ARCHIVE / "reproduced"))
OUT = OUTPUT_ROOT / "Fig1"
ORIGINAL_UNITS_PER_MM = 1598.2287 / 180.0

# Bounds include the panel letter and a small white margin, but exclude adjacent
# panels. Coordinates refer to the canonical editable SVG viewBox.
CROPS = {
    "a": (5.0, 10.0, 1590.0, 950.0),
    "b": (5.0, 960.0, 1590.0, 260.0),
    "c": (5.0, 1223.0, 415.0, 545.0),
    "d": (420.0, 1223.0, 1175.0, 545.0),
}
SVG_NS = "http://www.w3.org/2000/svg"


def replace_svg2_drop_shadows(root: ET.Element) -> None:
    """Replace SVG2 feDropShadow with widely supported SVG 1.1 primitives."""
    for parent in root.iter():
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag.rsplit("}", 1)[-1] != "feDropShadow":
                continue
            dx = child.get("dx", "0")
            dy = child.get("dy", "0")
            deviation = child.get("stdDeviation", "0")
            color = child.get("flood-color", "black")
            opacity = child.get("flood-opacity", "1")
            primitives = [
                ET.Element(
                    f"{{{SVG_NS}}}feGaussianBlur",
                    {"in": "SourceAlpha", "stdDeviation": deviation, "result": "blur"},
                ),
                ET.Element(
                    f"{{{SVG_NS}}}feOffset",
                    {"in": "blur", "dx": dx, "dy": dy, "result": "offsetBlur"},
                ),
                ET.Element(
                    f"{{{SVG_NS}}}feFlood",
                    {"flood-color": color, "flood-opacity": opacity, "result": "shadowColor"},
                ),
                ET.Element(
                    f"{{{SVG_NS}}}feComposite",
                    {"in": "shadowColor", "in2": "offsetBlur", "operator": "in", "result": "shadow"},
                ),
            ]
            merge = ET.Element(f"{{{SVG_NS}}}feMerge")
            ET.SubElement(merge, f"{{{SVG_NS}}}feMergeNode", {"in": "shadow"})
            ET.SubElement(merge, f"{{{SVG_NS}}}feMergeNode", {"in": "SourceGraphic"})
            primitives.append(merge)
            parent.remove(child)
            for offset, primitive in enumerate(primitives):
                parent.insert(index + offset, primitive)


def find_inkscape() -> str | None:
    candidate = shutil.which("inkscape")
    if candidate:
        return candidate
    app_binary = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")
    return str(app_binary) if app_binary.exists() else None


def export_derivatives(svg: Path, inkscape: str | None) -> None:
    if inkscape is None:
        return
    pdf = svg.with_suffix(".pdf")
    png = svg.with_suffix(".png")
    subprocess.run(
        [inkscape, str(svg), "--export-area-page", f"--export-filename={pdf}"],
        check=True,
    )
    subprocess.run(
        [inkscape, str(svg), "--export-area-page", "--export-dpi=600", f"--export-filename={png}"],
        check=True,
    )
    with Image.open(png) as image:
        image.convert("RGB").save(
            svg.with_suffix(".tiff"),
            dpi=(600, 600),
            compression="tiff_lzw",
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inkscape = find_inkscape()
    for panel, (x, y, width, height) in CROPS.items():
        tree = ET.parse(SOURCE)
        root = tree.getroot()
        replace_svg2_drop_shadows(root)
        root.set("viewBox", f"{x:g} {y:g} {width:g} {height:g}")
        root.set("width", f"{width / ORIGINAL_UNITS_PER_MM:.4f}mm")
        root.set("height", f"{height / ORIGINAL_UNITS_PER_MM:.4f}mm")
        output = OUT / f"figure1{panel}_standalone.svg"
        tree.write(output, encoding="utf-8", xml_declaration=True)
        export_derivatives(output, inkscape)
        print(output)


if __name__ == "__main__":
    main()
