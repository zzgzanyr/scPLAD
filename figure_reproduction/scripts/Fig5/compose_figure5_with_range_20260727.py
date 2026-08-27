#!/usr/bin/env python3
"""Append the text-free HSPA9 range panel to the editable Fig. 5 master."""

from __future__ import annotations

import copy
import re
from pathlib import Path

from lxml import etree


ARCHIVE = Path(__file__).resolve().parents[2]
WORK = ARCHIVE / "scripts"
BASE = ARCHIVE / "final_figures" / "Fig5" / "figure5_final_before_range_20260727.svg"
PANEL = (
    WORK
    / "figures_cross_cell_20260727"
    / "hspa9_range_fig5_style"
    / "figure5c_hspa9_expression_range_text_free.svg"
)
OUTPUT = ARCHIVE / "final_figures" / "Fig5" / "figure5_final.svg"

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
XLINK_NS = "http://www.w3.org/1999/xlink"
NS = {"svg": SVG_NS, "inkscape": INKSCAPE_NS}

WIDTH = 509.76
BASE_HEIGHT = 285.358736
PANEL_Y = 300.6
PANEL_HEIGHT = 101.52
FINAL_HEIGHT = 412.0
FONT_SIZE = 6.0
PANEL_SIZE = 8.0


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def iq(tag: str) -> str:
    return f"{{{INKSCAPE_NS}}}{tag}"


def prefix_ids(root: etree._Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for node in root.iter():
        node_id = node.get("id")
        if node_id:
            replacement = f"{prefix}_{node_id}"
            id_map[node_id] = replacement
            node.set("id", replacement)

    pattern = re.compile(r"url\(#([^)]+)\)")
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith("#") and value[1:] in id_map:
                node.set(key, f"#{id_map[value[1:]]}")
            else:
                node.set(
                    key,
                    pattern.sub(
                        lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                        value,
                    ),
                )


def add_text(
    parent: etree._Element,
    x: float,
    y: float,
    value: str,
    *,
    anchor: str = "start",
    panel: bool = False,
    rotate: float | None = None,
) -> etree._Element:
    node = etree.SubElement(
        parent,
        q("text"),
        x=f"{x:.4f}",
        y=f"{y:.4f}",
        **{
            "font-family": "Arial",
            "font-size": str(PANEL_SIZE if panel else FONT_SIZE),
            "font-weight": "bold" if panel else "normal",
            "font-style": "normal",
            "fill": "#000000",
            "stroke": "none",
            "stroke-width": "0",
            "text-anchor": anchor,
        },
    )
    if rotate is not None:
        node.set("transform", f"rotate({rotate:.1f} {x:.4f} {y:.4f})")
    node.text = value
    return node


def add_line(
    parent: etree._Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    colour: str,
    width: float,
) -> None:
    etree.SubElement(
        parent,
        q("line"),
        x1=f"{x1:.4f}",
        y1=f"{y1:.4f}",
        x2=f"{x2:.4f}",
        y2=f"{y2:.4f}",
        stroke=colour,
        **{"stroke-width": f"{width:.3f}"},
    )


def add_circle(
    parent: etree._Element,
    x: float,
    y: float,
    *,
    fill: str,
    stroke: str,
    radius: float = 1.8,
) -> None:
    etree.SubElement(
        parent,
        q("circle"),
        cx=f"{x:.4f}",
        cy=f"{y:.4f}",
        r=f"{radius:.3f}",
        fill=fill,
        stroke=stroke,
        **{"stroke-width": "0.55"},
    )


def main() -> None:
    tree = etree.parse(str(BASE))
    root = tree.getroot()
    root.set("height", f"{FINAL_HEIGHT * 25.4 / 72:.3f}mm")
    root.set("viewBox", f"0 0 {WIDTH:.6f} {FINAL_HEIGHT:.6f}")
    root.set("id", "figure5_hspa9_a_to_c_editable")

    graphics = root.find(".//svg:g[@id='graphics']", namespaces=NS)
    if graphics is None:
        raise RuntimeError("The Fig. 5 graphics layer was not found.")

    source = etree.parse(str(PANEL)).getroot()
    prefix_ids(source, "fig5_c")
    nested = etree.SubElement(
        graphics,
        q("svg"),
        id="fig5_c",
        x="0",
        y=f"{PANEL_Y:.6f}",
        width=f"{WIDTH:.6f}",
        height=f"{PANEL_HEIGHT:.6f}",
        viewBox=f"0 0 {WIDTH:.6f} {PANEL_HEIGHT:.6f}",
        preserveAspectRatio="none",
        overflow="visible",
    )
    nested.set(iq("label"), "fig5 c")
    for child in source:
        nested.append(copy.deepcopy(child))

    labels = etree.SubElement(
        root,
        q("g"),
        id="range_panel_labels",
    )
    labels.set(iq("groupmode"), "layer")
    labels.set(iq("label"), "Panel c labels")

    heading_y = 296.0
    add_text(labels, 7.654, heading_y, "c", panel=True)
    add_text(labels, 45.924, heading_y, "HSPA9 expression-range coverage")

    # Compact visual legend: each range bar is paired with its mean marker.
    add_line(labels, 326.0, 291.5, 326.0, 298.0, "#BFC6D0", 2.6)
    add_circle(labels, 326.0, 294.75, fill="#ffffff", stroke="#67717D")
    add_text(labels, 332.0, 296.6, "Observed")
    for x, colour in ((421.0, "#BE9FE5"), (424.2, "#F2C4C7"), (427.4, "#AFCBEA")):
        add_line(labels, x, 291.5, x, 298.0, colour, 1.65)
        add_circle(labels, x, 294.75, fill=colour, stroke=colour, radius=1.5)
    add_text(labels, 433.0, 296.6, "Generated")

    model_y = PANEL_Y + 7.0
    for x, label in (
        (99.34, "scPLAD | ERC₃₆ = 0.75"),
        (267.62, "STATE | ERC₃₆ = 0.82"),
        (435.91, "TxPert | ERC₃₆ = 0.14"),
    ):
        add_text(labels, x, model_y, label, anchor="middle")

    # Restore numeric y-axis ticks that were intentionally removed from the base.
    low, high = -4.2667671489715575, 1.7632689523696903
    axis_top, axis_bottom = PANEL_Y + 9.1368, PANEL_Y + 88.3224
    axis_x = 28.0368
    for value in (-4, -2, 0):
        y = axis_top + (high - value) / (high - low) * (axis_bottom - axis_top)
        add_line(labels, axis_x - 2.6, y, axis_x, y, "#000000", 0.6)
        add_text(labels, axis_x - 4.3, y + 2.0, str(value), anchor="end")

    add_text(
        labels,
        8.5,
        PANEL_Y + 49.0,
        "Expression change",
        anchor="middle",
        rotate=-90,
    )
    add_text(
        labels,
        267.62,
        PANEL_Y + 100.0,
        "Response genes ranked by observed effect",
        anchor="middle",
    )

    tree.write(
        str(OUTPUT),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
    )


if __name__ == "__main__":
    main()
