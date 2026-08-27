#!/usr/bin/env python3
"""Embed the user-edited semantic hub into the white, unframed Fig. 1 base."""

from __future__ import annotations

import copy
import re
import subprocess
from pathlib import Path

from lxml import etree


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "expression_glyph_variants"
    / "figure1_expression_dot_matrix_white_unframed.svg"
)
ASSET = (
    HERE
    / "expression_glyph_variants"
    / "semantic_context_vectors"
    / "single_cell_context_variant"
    / "single_semantic_hub_hybrid_7sources_single_cells_user_modified_20260713.svg"
)
OUTPUT = HERE / "composite_previews"
OUT_STEM = OUTPUT / "figure1_white_unframed_user_hub_preview"
PANEL_A_PNG = OUTPUT / "figure1a_white_unframed_user_hub_preview.png"
INKSCAPE = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")

SVG_NS = "http://www.w3.org/2000/svg"
SVG = f"{{{SVG_NS}}}"

# The crop excludes the source page whitespace while retaining every edited cell,
# helix, and the enlarged seven-source medallion.
ASSET_VIEWBOX = (22.0, 28.0, 126.0, 142.0)
TARGET = (680.0, 170.0, 345.0, 390.0)
OVERLAY = (675.0, 55.0, 350.0, 865.0)


def prefix_ids(root: etree._Element, prefix: str) -> None:
    mapping = {}
    for node in root.iter():
        element_id = node.get("id")
        if element_id:
            mapping[element_id] = f"{prefix}{element_id}"
            node.set("id", mapping[element_id])

    fragment_pattern = re.compile(r"#([A-Za-z_][\w:.-]*)")
    event_pattern = re.compile(r"(?<![\w-])([A-Za-z_][\w:-]*)(?=\.)")

    for node in root.iter():
        for key, value in list(node.attrib.items()):
            updated = fragment_pattern.sub(
                lambda match: f"#{mapping.get(match.group(1), match.group(1))}",
                value,
            )
            updated = event_pattern.sub(
                lambda match: mapping.get(match.group(1), match.group(1)),
                updated,
            )
            if updated != value:
                node.set(key, updated)


def add_title(parent: etree._Element) -> None:
    title = etree.SubElement(parent, f"{SVG}text")
    title.set("x", "850")
    title.set("y", "88")
    title.set("fill", "#174A8B")
    title.set("font-family", "Arial, Helvetica, sans-serif")
    title.set("font-size", "25")
    title.set("font-weight", "700")
    title.set("text-anchor", "middle")

    line1 = etree.SubElement(title, f"{SVG}tspan")
    line1.set("x", "850")
    line1.set("dy", "0")
    line1.text = "Cross-cell-line"
    line2 = etree.SubElement(title, f"{SVG}tspan")
    line2.set("x", "850")
    line2.set("dy", "30")
    line2.text = "knowledge sharing"


def build() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(SOURCE), parser)
    root = tree.getroot()

    asset_tree = etree.parse(str(ASSET), parser)
    asset_root = copy.deepcopy(asset_tree.getroot())
    prefix_ids(asset_root, "embedded_hub_")

    layer = etree.Element(f"{SVG}g")
    layer.set("id", "figure1a_user_semantic_hub_overlay")

    x, y, width, height = OVERLAY
    cover = etree.SubElement(layer, f"{SVG}rect")
    cover.set("x", str(x))
    cover.set("y", str(y))
    cover.set("width", str(width))
    cover.set("height", str(height))
    cover.set("fill", "#FFFFFF")
    add_title(layer)

    tx, ty, tw, th = TARGET
    vx, vy, vw, vh = ASSET_VIEWBOX
    scale = min(tw / vw, th / vh)
    placed_width = vw * scale
    placed_height = vh * scale
    origin_x = tx + (tw - placed_width) / 2.0 - vx * scale
    origin_y = ty + (th - placed_height) / 2.0 - vy * scale

    clip = etree.SubElement(layer, f"{SVG}clipPath")
    clip.set("id", "figure1a_user_semantic_hub_clip")
    clip.set("clipPathUnits", "userSpaceOnUse")
    clip_rect = etree.SubElement(clip, f"{SVG}rect")
    clip_rect.set("x", str(tx))
    clip_rect.set("y", str(ty))
    clip_rect.set("width", str(tw))
    clip_rect.set("height", str(th))

    clipped = etree.SubElement(layer, f"{SVG}g")
    clipped.set("clip-path", "url(#figure1a_user_semantic_hub_clip)")
    nested = etree.SubElement(clipped, f"{SVG}g")
    nested.set("id", "figure1a_user_semantic_hub")
    nested.set("transform", f"translate({origin_x:.6f},{origin_y:.6f}) scale({scale:.9f})")
    for child in asset_root:
        nested.append(copy.deepcopy(child))

    root.append(layer)
    out_svg = Path(f"{OUT_STEM}.svg")
    tree.write(str(out_svg), encoding="UTF-8", xml_declaration=True, pretty_print=False)
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
