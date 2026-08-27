#!/usr/bin/env python3
"""Replace Fig. 1c's tiny prior icons with compact text-based source modules."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lxml import etree


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "composite_previews" / "figure1_white_unframed_user_hub_enlarged120.svg"
OUT_STEM = HERE / "composite_previews" / "figure1_white_unframed_user_hub_enlarged120_textprior"
PANEL_C_PNG = HERE / "composite_previews" / "figure1c_textprior_preview.png"
INKSCAPE = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")

SVG_NS = "http://www.w3.org/2000/svg"
SVG = f"{{{SVG_NS}}}"
FONT = "Arial, Helvetica, sans-serif"
INK = "#17212B"
GREEN = "#3B7F32"


def rect(parent, x, y, width, height, *, fill, stroke="none", stroke_width=0, radius=0):
    node = etree.SubElement(parent, f"{SVG}rect")
    node.set("x", str(x))
    node.set("y", str(y))
    node.set("width", str(width))
    node.set("height", str(height))
    if radius:
        node.set("rx", str(radius))
        node.set("ry", str(radius))
    node.set("fill", fill)
    node.set("stroke", stroke)
    node.set("stroke-width", str(stroke_width))
    return node


def text_node(parent, x, y, value, *, size=16, weight=400, anchor="start", fill=INK, italic=False):
    node = etree.SubElement(parent, f"{SVG}text")
    node.set("x", str(x))
    node.set("y", str(y))
    node.set("font-family", FONT)
    node.set("font-size", str(size))
    node.set("font-weight", str(weight))
    node.set("text-anchor", anchor)
    node.set("fill", fill)
    if italic:
        node.set("font-style", "italic")
    node.text = value
    return node


def two_line_text(parent, x, y, first, second, *, size=16, weight=400, anchor="middle", gap=20):
    node = etree.SubElement(parent, f"{SVG}text")
    node.set("x", str(x))
    node.set("y", str(y))
    node.set("font-family", FONT)
    node.set("font-size", str(size))
    node.set("font-weight", str(weight))
    node.set("text-anchor", anchor)
    node.set("fill", INK)
    line1 = etree.SubElement(node, f"{SVG}tspan")
    line1.set("x", str(x))
    line1.text = first
    line2 = etree.SubElement(node, f"{SVG}tspan")
    line2.set("x", str(x))
    line2.set("dy", str(gap))
    line2.text = second
    return node


def arrow(parent, x1, y1, x2, y2, *, color=INK, width=2.0, head=7.0):
    line = etree.SubElement(parent, f"{SVG}line")
    line.set("x1", str(x1))
    line.set("y1", str(y1))
    line.set("x2", str(x2))
    line.set("y2", str(y2))
    line.set("stroke", color)
    line.set("stroke-width", str(width))
    line.set("stroke-linecap", "round")

    if abs(x2 - x1) >= abs(y2 - y1):
        points = [(x2, y2), (x2 - head, y2 - head * 0.55), (x2 - head, y2 + head * 0.55)]
    else:
        points = [(x2, y2), (x2 - head * 0.55, y2 - head), (x2 + head * 0.55, y2 - head)]
    tip = etree.SubElement(parent, f"{SVG}polygon")
    tip.set("points", " ".join(f"{x},{y}" for x, y in points))
    tip.set("fill", color)


def source_row(parent, y, label, sources, fill, stroke):
    rect(parent, 69, y, 307, 35, fill=fill, stroke=stroke, stroke_width=1.2, radius=7)
    rect(parent, 69, y, 7, 35, fill=stroke, radius=7)
    text_node(parent, 84, y + 14.5, label, size=13.2, weight=700)
    text_node(parent, 84, y + 29.0, sources, size=11.8, weight=400, fill="#43515C")


def add_panel(root: etree._Element) -> None:
    panel = etree.SubElement(root, f"{SVG}g")
    panel.set("id", "figure1c_text_based_prior_sources")

    # Cover only the interior so the original green panel border and panel label remain.
    rect(panel, 35.0, 1295.0, 375.0, 504.0, fill="#FFFFFF")

    two_line_text(
        panel,
        222.5,
        1325.0,
        "Biological priors encode",
        "perturbation semantics",
        size=24,
        weight=700,
        gap=27,
    )

    rect(panel, 104, 1375, 237, 46, fill="#FFFFFF", stroke=GREEN, stroke_width=1.4, radius=12)
    gene = etree.SubElement(panel, f"{SVG}text")
    gene.set("x", "222.5")
    gene.set("y", "1405")
    gene.set("font-family", FONT)
    gene.set("font-size", "18")
    gene.set("text-anchor", "middle")
    gene.set("fill", INK)
    gene.text = "Perturbation gene "
    symbol = etree.SubElement(gene, f"{SVG}tspan")
    symbol.set("font-style", "italic")
    symbol.set("font-weight", "700")
    symbol.text = "g"

    arrow(panel, 222.5, 1423, 222.5, 1443, color=GREEN, width=2.1, head=7)

    rect(panel, 56, 1446, 333, 210, fill="#FFFFFF", stroke="#8AB07C", stroke_width=1.5, radius=12)
    text_node(panel, 222.5, 1475, "Multi-source public biological priors", size=17, weight=700, anchor="middle")
    source_row(panel, 1486, "Functional annotation", "GO · Reactome", "#EEF6E9", "#8AB07C")
    source_row(panel, 1525, "Protein sequence", "ESM3", "#EDF4FC", "#74A5D8")
    source_row(panel, 1564, "Molecular networks", "STRING/PPI · GRN · OmniPath", "#EAF6F3", "#299D8F")
    source_row(panel, 1603, "Protein complexes", "CORUM", "#FCEDEA", "#E66D50")

    arrow(panel, 222.5, 1658, 222.5, 1676, color="#607080", width=1.8, head=6.5)
    text_node(panel, 222.5, 1700, "Concatenated prior vector  r_g  (2336 d)", size=15.5, weight=600, anchor="middle")

    rect(panel, 58, 1720, 126, 57, fill="#FFFFFF", stroke="#8AB07C", stroke_width=1.4, radius=10)
    two_line_text(panel, 121, 1743, "Prior", "encoder", size=17, weight=700, gap=19)
    arrow(panel, 192, 1748.5, 231, 1748.5, color="#44515D", width=2.0, head=7)

    rect(panel, 239, 1713, 147, 71, fill="#F7FAFD", stroke="#74A5D8", stroke_width=1.4, radius=10)
    text_node(panel, 312.5, 1736, "Condition embedding", size=15.5, weight=600, anchor="middle")
    text_node(panel, 312.5, 1757, "e_c", size=17, weight=700, anchor="middle", italic=True)
    vector_colors = (
        ("#8AB07C", "#5F8E54"),
        ("#74A5D8", "#4F79A8"),
        ("#299D8F", "#24786F"),
        ("#E66D50", "#B84F3A"),
    )
    start_x = 275.0
    for index, (fill, stroke) in enumerate(vector_colors):
        rect(
            panel,
            start_x + index * 20,
            1763,
            15,
            15,
            fill=fill,
            stroke=stroke,
            stroke_width=1.0,
            radius=3,
        )


def export(svg_path: Path) -> None:
    subprocess.run(
        [str(INKSCAPE), str(svg_path), "--export-type=pdf", f"--export-filename={OUT_STEM}.pdf", "--export-area-page"],
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
            f"--export-filename={PANEL_C_PNG}",
            "--export-area=0:525:180:764",
            "--export-width=1200",
        ],
        check=True,
    )


def main() -> None:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(SOURCE), parser)
    root = tree.getroot()
    add_panel(root)
    out_svg = Path(f"{OUT_STEM}.svg")
    tree.write(str(out_svg), encoding="UTF-8", xml_declaration=True, pretty_print=False)
    export(out_svg)
    print(out_svg)


if __name__ == "__main__":
    main()
