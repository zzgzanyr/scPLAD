#!/usr/bin/env python3
"""Apply the 180-mm publication layout to the current fine-tuned Figure 2.

The source SVG is not overwritten. Geometry inside each panel is preserved;
only panel viewports, page size, and typography are standardized.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figure2_inkscape_uniform_exact_equalplots.svg"
OUT_DIR = HERE / "publication_180mm_6pt"
OUTPUT = OUT_DIR / "figure2_180mm_6pt.svg"
INKSCAPE = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
Q = lambda tag: f"{{{SVG_NS}}}{tag}"

# 180 mm converted to PostScript points. Margins and gaps are exactly 3 mm.
MM_TO_PT = 72.0 / 25.4
PAGE_W = 180.0 * MM_TO_PT
SPACE = 3.0 * MM_TO_PT

PANEL_A_W, PANEL_A_H = 470.25, 130.05
# Panel b's artists extend 8.685 pt left and 16.709 pt right of its original
# Matplotlib viewport. Expand the viewBox instead of scaling the panel so the
# right-hand CSA example remains complete at the final 6-pt type size.
PANEL_B_W, PANEL_B_H = 455.954, 158.40
PANEL_C_W, PANEL_C_H = PAGE_W - 2 * SPACE - SPACE - 198.0, 141.30
PANEL_D_W, PANEL_D_H = 198.0, 141.30

Y_A = SPACE
Y_B = Y_A + PANEL_A_H + SPACE
Y_BOTTOM = Y_B + PANEL_B_H + SPACE
PAGE_H = Y_BOTTOM + PANEL_C_H + SPACE

X_A = (PAGE_W - PANEL_A_W) / 2
X_B = (PAGE_W - PANEL_B_W) / 2
X_C = SPACE
X_D = X_C + PANEL_C_W + SPACE


def parse_style(value: str | None) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for item in (value or "").split(";"):
        if ":" in item:
            key, val = item.split(":", 1)
            declarations[key.strip()] = val.strip()
    return declarations


def format_style(declarations: dict[str, str]) -> str:
    return "; ".join(f"{key}: {value}" for key, value in declarations.items())


def set_text_style(element: ET.Element, *, size: float, bold: bool) -> None:
    element.set("font-family", "Arial")
    element.set("font-size", f"{size:g}")
    element.set("font-weight", "bold" if bold else "normal")
    element.set("font-style", "normal")
    element.set("fill", "#000000")

    style = parse_style(element.get("style"))
    style["font-family"] = "Arial"
    style["font-size"] = f"{size:g}px"
    style["font-weight"] = "700" if bold else "400"
    style["font-style"] = "normal"
    style["fill"] = "#000000"
    element.set("style", format_style(style))


def direct_nested_svg(panel: ET.Element) -> ET.Element:
    return next(child for child in panel if child.tag == Q("svg"))


def direct_panel_label(panel: ET.Element, letter: str) -> ET.Element:
    return next(
        child
        for child in panel
        if child.tag == Q("text") and (child.text or "").strip() == letter
    )


def remove_erc_union_backgrounds(root: ET.Element) -> None:
    """Keep only the interval intersection in the ERC schematic.

    The pale rectangles originally visualized the union of the true and
    predicted intervals.  ERC uses the overlap length as its numerator, so
    retaining only the green overlap makes the visual encoding unambiguous.
    """
    union_ids = {"path3", "path14", "path25"}
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in list(root.iter(Q("path"))):
        if element.get("id") in union_ids:
            parents[element].remove(element)


def add_erc_quantile_labels(panel: ET.Element) -> None:
    """Mark the 10th and 90th percentiles that define each true interval."""
    nested = direct_nested_svg(panel)
    label_specs = (
        (39.91849, "q10"),
        (101.448906, "q90"),
        (196.66849, "q10"),
        (258.198906, "q90"),
        (353.41849, "q10"),
        (414.948906, "q90"),
    )
    for index, (x, label) in enumerate(label_specs, start=1):
        text = ET.SubElement(
            nested,
            Q("text"),
            {
                "id": f"erc_quantile_{index}",
                "x": f"{x:.6f}",
                "y": "31.500000",
                "text-anchor": "middle",
                "fill": "#000000",
            },
        )
        text.text = label


def set_panel_viewport(
    panel: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    crop_viewbox_width: bool = False,
    viewbox_x: float | None = None,
) -> None:
    nested = direct_nested_svg(panel)
    nested.set("x", f"{x:.6f}")
    nested.set("y", f"{y:.6f}")
    nested.set("width", f"{width:.6f}")
    nested.set("height", f"{height:.6f}")
    if crop_viewbox_width or viewbox_x is not None:
        viewbox = [float(value) for value in nested.get("viewBox", f"0 0 {width} {height}").split()]
        if viewbox_x is not None:
            viewbox[0] = viewbox_x
        viewbox[2] = width
        nested.set("viewBox", " ".join(f"{value:.6f}" for value in viewbox))


def build() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    remove_erc_union_backgrounds(root)
    panels = {
        letter: next(element for element in root.iter(Q("g")) if element.get("id") == f"panel_{letter}_exact")
        for letter in "abcd"
    }
    add_erc_quantile_labels(panels["a"])

    root.set("width", "180mm")
    root.set("height", f"{PAGE_H / MM_TO_PT:.6f}mm")
    root.set("viewBox", f"0 0 {PAGE_W:.6f} {PAGE_H:.6f}")

    set_panel_viewport(panels["a"], x=X_A, y=Y_A, width=PANEL_A_W, height=PANEL_A_H)
    set_panel_viewport(
        panels["b"],
        x=X_B,
        y=Y_B,
        width=PANEL_B_W,
        height=PANEL_B_H,
        viewbox_x=-8.685,
    )
    # Panel c contains enough unused right-side whitespace to crop its viewport
    # without scaling either plotting area or its text.
    set_panel_viewport(
        panels["c"],
        x=X_C,
        y=Y_BOTTOM,
        width=PANEL_C_W,
        height=PANEL_C_H,
        crop_viewbox_width=True,
    )
    set_panel_viewport(panels["d"], x=X_D, y=Y_BOTTOM, width=PANEL_D_W, height=PANEL_D_H)

    for element in root.iter():
        if element.tag in {Q("text"), Q("tspan")}:
            set_text_style(element, size=6, bold=False)

    label_positions = {
        "a": (SPACE, Y_A + 8.0),
        "b": (SPACE, Y_B + 8.0),
        "c": (X_C, Y_BOTTOM + 8.0),
        "d": (X_D, Y_BOTTOM + 8.0),
    }
    for letter, panel in panels.items():
        label = direct_panel_label(panel, letter)
        label.set("x", f"{label_positions[letter][0]:.6f}")
        label.set("y", f"{label_positions[letter][1]:.6f}")
        # Keep panel letters consistent with the manuscript-wide 6-pt,
        # regular-weight annotation system.
        set_text_style(label, size=6, bold=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)


def export() -> None:
    pdf = OUTPUT.with_suffix(".pdf")
    png = OUTPUT.with_name(f"{OUTPUT.stem}_preview.png")
    subprocess.run([str(INKSCAPE), str(OUTPUT), f"--export-filename={pdf}", "--export-area-page"], check=True)
    subprocess.run(
        [str(INKSCAPE), str(OUTPUT), f"--export-filename={png}", "--export-area-page", "--export-dpi=180"],
        check=True,
    )


def main() -> None:
    build()
    export()


if __name__ == "__main__":
    main()
