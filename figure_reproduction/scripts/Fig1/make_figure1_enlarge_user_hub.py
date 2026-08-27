#!/usr/bin/env python3
"""Create a Fig. 1 preview with the complete semantic-context hub enlarged."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lxml import etree


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "composite_previews" / "figure1_white_unframed_user_hub_preview.svg"
OUT_STEM = HERE / "composite_previews" / "figure1_white_unframed_user_hub_enlarged120"
PANEL_A_PNG = HERE / "composite_previews" / "figure1a_white_unframed_user_hub_enlarged120.png"
INKSCAPE = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")

SCALE = 1.20
# Center of the complete imported hub in the full Fig. 1 viewBox coordinates.
CENTER = (876.6, 536.2)
OBJECT_IDS = (
    "g1",
    "prior_context_dna_bridge_inverted_a",
    "prior_context_dna_bridge_inverted_b",
    "prior_context_dna_bridge_inverted_c",
)


def first(root: etree._Element, element_id: str) -> etree._Element:
    matches = root.xpath(f'//*[@id="{element_id}"]')
    if not matches:
        raise RuntimeError(f"Missing SVG object: {element_id}")
    return matches[0]


def build() -> Path:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(SOURCE), parser)
    root = tree.getroot()
    objects = [first(root, element_id) for element_id in OBJECT_IDS]
    if any(node.getparent() is not root for node in objects):
        raise RuntimeError("Hub objects are no longer direct children of the SVG root")

    insertion_index = min(root.index(node) for node in objects)
    wrapper = etree.Element("{http://www.w3.org/2000/svg}g")
    wrapper.set("id", "enlarged_semantic_context_hub")
    cx, cy = CENTER
    wrapper.set(
        "transform",
        f"translate({cx:.4f},{cy:.4f}) scale({SCALE:.4f}) "
        f"translate({-cx:.4f},{-cy:.4f})",
    )
    root.insert(insertion_index, wrapper)
    for node in sorted(objects, key=root.index):
        wrapper.append(node)

    out_svg = Path(f"{OUT_STEM}.svg")
    tree.write(
        str(out_svg),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=False,
    )
    return out_svg


def export(svg_path: Path) -> None:
    subprocess.run(
        [
            str(INKSCAPE),
            str(svg_path),
            "--export-type=pdf",
            f"--export-filename={OUT_STEM}.pdf",
            "--export-area-page",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(INKSCAPE),
            str(svg_path),
            "--export-type=png",
            f"--export-filename={OUT_STEM}.png",
            "--export-area-page",
            "--export-width=2200",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(INKSCAPE),
            str(svg_path),
            f"--export-filename={PANEL_A_PNG}",
            "--export-area=0:0:1598.2287:970",
            "--export-width=2200",
        ],
        check=True,
    )


def main() -> None:
    out_svg = build()
    export(out_svg)
    print(out_svg)


if __name__ == "__main__":
    main()
