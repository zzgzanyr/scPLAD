#!/usr/bin/env python3
"""Build the white-background, framed dot-matrix version of Fig. 1."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lxml import etree

from make_figure1_expression_glyph_variants import INKSCAPE, SOURCE, SVG_NS, dot_matrix


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "expression_glyph_variants"
OUT_SVG = OUTPUT / "figure1_expression_dot_matrix_white_unframed.svg"
OUT_PDF = OUTPUT / "figure1_expression_dot_matrix_white_unframed.pdf"
OUT_PNG = OUTPUT / "figure1_expression_dot_matrix_white_unframed.png"

SVG = f"{{{SVG_NS}}}"

PANEL_BACKGROUND_IDS = {
    "fig1bcd_rect11",   # panel b
    "fig1bcd_rect188",  # panel c
    "fig1bcd_rect254",  # panel d
}

PANEL_A_MATRIX_IDS = {
    "fig1a_g293",
    "fig1a_g469",
    "fig1a_g578",
    "fig1a_g919",
    "source_style_expression_matrix_variant",
}

ALL_EXPRESSION_MATRIX_IDS = PANEL_A_MATRIX_IDS | {
    "fig1bcd_heatmap_blue_11",
    "fig1bcd_heatmap_purple_12",
    "fig1bcd_heatmap_blue_31",
    "fig1bcd_heatmap_purple_42",
}


def first(root: etree._Element, element_id: str) -> etree._Element:
    matches = root.xpath(f'//*[@id="{element_id}"]')
    if not matches:
        raise RuntimeError(f"SVG element not found: {element_id}")
    return matches[0]


def make_backgrounds_white(root: etree._Element) -> None:
    # The imported panel-a plate slightly exceeds the SVG page on the right,
    # so using its stroke would clip the frame. Keep it visually transparent
    # and add a page-coordinate frame aligned with panels b-d instead.
    panel_a = first(root, "fig1a_rect13")
    panel_a.set("fill", "none")
    panel_a.set("stroke", "none")
    panel_a.set("style", "display:inline;fill:none;stroke:none")

    panel_a_group = first(root, "panel_a_group")
    panel_a_frame = etree.Element(f"{SVG}rect")
    panel_a_frame.set("id", "panel_a_outer_frame_white")
    panel_a_frame.set("x", "32")
    panel_a_frame.set("y", "48")
    panel_a_frame.set("width", "1540")
    panel_a_frame.set("height", "910")
    panel_a_frame.set("rx", "15")
    panel_a_frame.set("ry", "15")
    panel_a_frame.set("fill", "none")
    panel_a_frame.set("stroke", "#1f4fa7")
    panel_a_frame.set("stroke-width", "1.6")
    # A lower-page white plate starts at y=921.8 and would hide the bottom
    # edge. Draw the transparent frame immediately after that plate.
    lower_page_plate = first(root, "fig1bcd_rect10")
    root.insert(root.index(lower_page_plate) + 1, panel_a_frame)

    # Remove the two colored backdrop blocks inside panel a. Their content is
    # still separated by spacing and headings, while the new outer frame keeps
    # the whole panel aligned with b-d.
    for element_id in ("fig1a_rect14", "fig1a_rect41"):
        block = first(root, element_id)
        block.set("fill", "#ffffff")
        block.set("fill-opacity", "1")
        block.set("opacity", "1")
        block.set("stroke", "none")
        block.set("style", "display:inline;fill:#ffffff;stroke:none")

    # Panels b-d retain their method-family outline colors but lose the tinted
    # gradient fills, giving the full figure one continuous white canvas.
    for element_id in PANEL_BACKGROUND_IDS:
        panel = first(root, element_id)
        panel.set("fill", "#ffffff")
        panel.set("fill-opacity", "1")
        panel.set("opacity", "1")
        panel.set("style", "fill:#ffffff")


def add_panel_a_matrix_frames(root: etree._Element) -> None:
    for group_id in PANEL_A_MATRIX_IDS:
        group = first(root, group_id)
        circles = [child for child in group if etree.QName(child).localname == "circle"]
        if not circles:
            raise RuntimeError(f"No expression dots found in {group_id}")

        centers_x = sorted({round(float(circle.get("cx")), 5) for circle in circles})
        centers_y = sorted({round(float(circle.get("cy")), 5) for circle in circles})
        step_x = min((b - a for a, b in zip(centers_x, centers_x[1:])), default=16.0)
        step_y = min((b - a for a, b in zip(centers_y, centers_y[1:])), default=16.0)
        x = centers_x[0] - step_x / 2
        y = centers_y[0] - step_y / 2
        width = centers_x[-1] - centers_x[0] + step_x
        height = centers_y[-1] - centers_y[0] + step_y

        frame = etree.Element(f"{SVG}rect")
        frame.set("id", f"{group_id}_expression_frame")
        frame.set("x", f"{x:.4f}")
        frame.set("y", f"{y:.4f}")
        frame.set("width", f"{width:.4f}")
        frame.set("height", f"{height:.4f}")
        frame.set("rx", "0")
        frame.set("fill", "#ffffff")
        frame.set("stroke", "#8aa6d8")
        frame.set("stroke-width", "0.65")
        group.insert(0, frame)


def remove_expression_matrix_frames(root: etree._Element) -> None:
    """Remove rigid heatmap frames while preserving every expression dot."""
    for group_id in ALL_EXPRESSION_MATRIX_IDS:
        group = first(root, group_id)
        for child in list(group):
            if etree.QName(child).localname != "rect":
                continue
            fill = child.get("fill", "")
            style = child.get("style", "")
            if (
                child.get("id", "").endswith("_expression_frame")
                or fill == "none"
                or "fill:none" in style
            ):
                group.remove(child)


def export(tree: etree._ElementTree) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tree.write(str(OUT_SVG), encoding="UTF-8", xml_declaration=True, pretty_print=False)
    subprocess.run(
        [
            str(INKSCAPE),
            str(OUT_SVG),
            f"--export-filename={OUT_PDF}",
            "--export-area-page",
            "--export-page=1",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(INKSCAPE),
            str(OUT_SVG),
            f"--export-filename={OUT_PNG}",
            "--export-area-page",
            "--export-page=1",
            "--export-dpi=180",
        ],
        check=True,
    )


def main() -> None:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(SOURCE), parser)
    root = tree.getroot()
    dot_matrix(root)
    make_backgrounds_white(root)
    remove_expression_matrix_frames(root)
    export(tree)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
