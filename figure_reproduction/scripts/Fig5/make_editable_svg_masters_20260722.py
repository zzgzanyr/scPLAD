#!/usr/bin/env python3
"""Create self-contained, Inkscape-editable SVG masters for Figures 2 and 4.

Unlike the earlier PDF-to-SVG exports, this script composes native vector SVG
panels directly.  Their text remains SVG text and can be edited in Inkscape.
The source SVG panels and their matching PDF panels are produced from the same
plotting scripts; PDFs remain the archival publication components.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_editable_svg_masters_20260722"
SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, "inkscape": INKSCAPE_NS, "xlink": XLINK_NS}

NORMAL_SIZE = 6.0
PANEL_SIZE = 8.0
PANEL_LETTERS = set("abcdefgh")


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def iq(tag: str) -> str:
    return f"{{{INKSCAPE_NS}}}{tag}"


def tag_name(node: etree._Element) -> str:
    try:
        return etree.QName(node).localname
    except ValueError:
        return ""


def parse_viewbox(root: etree._Element) -> tuple[float, float, float, float]:
    values = [float(value) for value in root.get("viewBox").split()]
    if len(values) != 4:
        raise ValueError(f"Missing SVG viewBox: {root.base}")
    return tuple(values)  # type: ignore[return-value]


def text_value(node: etree._Element) -> str:
    return "".join(node.itertext()).strip().replace("\n", " ")


def prefix_ids(root: etree._Element, prefix: str) -> None:
    """Avoid clashes among marker, clipPath and glyph identifiers."""
    id_map: dict[str, str] = {}
    for node in root.iter():
        node_id = node.get("id")
        if node_id:
            replacement = f"{prefix}_{node_id}"
            id_map[node_id] = replacement
            node.set("id", replacement)
    url_pattern = re.compile(r"url\(#([^)]+)\)")
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith("#") and value[1:] in id_map:
                node.set(key, f"#{id_map[value[1:]]}")
            else:
                node.set(
                    key,
                    url_pattern.sub(
                        lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                        value,
                    ),
                )


def normalize_text(
    root: etree._Element,
    scale: float,
    panel_label_map: dict[str, str] | None = None,
    left_align_hspa_title: bool = False,
) -> None:
    """Keep labels editable while enforcing the manuscript typography."""
    for node in root.iter():
        if tag_name(node) != "text":
            continue
        value = text_value(node)
        if panel_label_map and value in panel_label_map:
            node.text = panel_label_map[value]
            value = panel_label_map[value]
        is_panel = value in PANEL_LETTERS
        size = (PANEL_SIZE if is_panel else NORMAL_SIZE) / scale
        anchor = node.get("text-anchor")
        style = node.get("style", "")
        match = re.search(r"text-anchor\s*:\s*([^;]+)", style)
        if match and not anchor:
            anchor = match.group(1).strip()
        node.set("font-family", "Arial")
        node.set("font-size", f"{size:.4f}")
        node.set("font-weight", "bold" if is_panel else "normal")
        node.set("font-style", "normal")
        node.set("fill", "#000000")
        if anchor:
            node.set("text-anchor", anchor)
        node.attrib.pop("style", None)
        if left_align_hspa_title and value == "HSPA9 external causal target recovery":
            # Align the case-study heading with panel b rather than centring it.
            node.set("x", "40")
            node.set("text-anchor", "start")
            node.attrib.pop("transform", None)


def append_panel(
    parent: etree._Element,
    source_path: Path,
    *,
    panel_id: str,
    x: float,
    y: float,
    scale: float = 1.0,
    panel_label_map: dict[str, str] | None = None,
    left_align_hspa_title: bool = False,
) -> tuple[float, float]:
    """Embed one native SVG panel without external links or rasterization."""
    source = etree.parse(str(source_path)).getroot()
    x0, y0, width, height = parse_viewbox(source)
    prefix_ids(source, panel_id)
    normalize_text(source, scale, panel_label_map, left_align_hspa_title)
    nested = etree.SubElement(
        parent,
        q("svg"),
        id=panel_id,
        x=f"{x:.6f}",
        y=f"{y:.6f}",
        width=f"{width * scale:.6f}",
        height=f"{height * scale:.6f}",
        viewBox=f"{x0:.6f} {y0:.6f} {width:.6f} {height:.6f}",
        preserveAspectRatio="none",
        overflow="visible",
    )
    nested.set(iq("label"), panel_id.replace("_", " "))
    for child in source:
        nested.append(copy.deepcopy(child))
    return width * scale, height * scale


def add_text(
    parent: etree._Element,
    x: float,
    y: float,
    value: str,
    *,
    panel: bool = False,
    anchor: str = "start",
) -> None:
    node = etree.SubElement(
        parent,
        q("text"),
        x=f"{x:.6f}",
        y=f"{y:.6f}",
        **{
            "font-family": "Arial",
            "font-size": str(PANEL_SIZE if panel else NORMAL_SIZE),
            "font-weight": "bold" if panel else "normal",
            "font-style": "normal",
            "fill": "#000000",
            "text-anchor": anchor,
        },
    )
    node.text = value


def add_circle(parent: etree._Element, x: float, y: float, colour: str) -> None:
    etree.SubElement(parent, q("circle"), cx=f"{x:.6f}", cy=f"{y:.6f}", r="2", fill=colour)


def master(width: float, height: float, filename: str) -> etree._ElementTree:
    root = etree.Element(
        q("svg"),
        nsmap=NSMAP,
        width="180mm",
        height=f"{height * 25.4 / 72:.3f}mm",
        viewBox=f"0 0 {width:.6f} {height:.6f}",
        version="1.1",
        id=filename.removesuffix(".svg"),
    )
    root.set(iq("document-units"), "pt")
    return etree.ElementTree(root)


def make_figure2() -> None:
    width, base_height, gap = 510.236220, 463.765748, 4.0
    umap_height = 95.04
    tree = master(width, base_height + gap + umap_height, "figure2_editable_master.svg")
    root = tree.getroot()
    graphics = etree.SubElement(root, q("g"), id="graphics")
    graphics.set(iq("groupmode"), "layer")
    graphics.set(iq("label"), "Vector panels")
    append_panel(
        graphics,
        ROOT / "figure2_vector_no_text" / "publication_180mm_6pt" / "figure2_180mm_6pt.svg",
        panel_id="fig2_a_to_d", x=0, y=0,
    )
    append_panel(
        graphics,
        ROOT / "figures_fig2_pastel_additions_20260721" / "figure2_patchae_vae_shared_umap.svg",
        panel_id="fig2_e_umap", x=0, y=base_height + gap,
    )
    labels = etree.SubElement(root, q("g"), id="labels")
    labels.set(iq("groupmode"), "layer")
    labels.set(iq("label"), "Unified editable labels")
    add_text(labels, 8.503937, base_height + gap + 11, "e", panel=True)
    # The compact UMAP legend is added here rather than baked into the panel.
    for x, y, colour, value in [
        (438, 486, "#77729A", "Control"),
        (438, 501, "#BE9FE5", "HSPA9"),
        (438, 516, "#6FA1D9", "NCBP2"),
        (438, 531, "#EAA8AE", "SLC39A9"),
    ]:
        add_circle(labels, x, y, colour)
        add_text(labels, x + 5, y + 2, value)
    tree.write(str(OUT / "figure2_editable_master.svg"), encoding="utf-8", xml_declaration=True, pretty_print=True)


def make_figure4() -> None:
    width, scale, shift, gap = 509.76, 0.92, 10.0, 5.67
    top_path = ROOT / "figures_cross_cell_20260718" / "panel_a_overall_mean_metrics" / "figure4a_cross_cell_overall_mean_metrics.svg"
    middle_path = ROOT / "figures_cross_cell_20260718" / "panels_b_g_quantitative_mean_txpert" / "figure_cross_cell_coverage_b_e_mean_txpert.svg"
    bottom_path = ROOT / "figures_cross_cell_20260718" / "panels_f_g_hspa9" / "rendered_20260721" / "figure4f_hspa9_external_targets.svg"
    csa_path = ROOT / "figures_cross_cell_20260718" / "panels_f_g_hspa9" / "csa_heatmap_20260721" / "figure4g_hspa9_csa_heatmap.svg"
    _, top_height = append_dimensions(top_path, scale)
    _, middle_height = append_dimensions(middle_path, scale)
    _, bottom_height = append_dimensions(bottom_path, scale)
    _, csa_height = append_dimensions(csa_path, width / 510.264)
    base_height = top_height + gap + middle_height + gap + bottom_height
    total_height = base_height + 4.0 + csa_height
    tree = master(width, total_height, "figure4_editable_master.svg")
    root = tree.getroot()
    graphics = etree.SubElement(root, q("g"), id="graphics")
    graphics.set(iq("groupmode"), "layer")
    graphics.set(iq("label"), "Vector panels")
    append_panel(graphics, top_path, panel_id="fig4_a", x=shift, y=0, scale=scale)
    append_panel(graphics, middle_path, panel_id="fig4_b_to_e", x=shift, y=top_height + gap, scale=scale)
    append_panel(graphics, bottom_path, panel_id="fig4_f", x=shift, y=top_height + gap + middle_height + gap, scale=scale)
    append_panel(graphics, csa_path, panel_id="fig4_g", x=0, y=base_height + 4.0, scale=width / 510.264)
    tree.write(str(OUT / "figure4_editable_master.svg"), encoding="utf-8", xml_declaration=True, pretty_print=True)


def figure4_sources() -> tuple[Path, Path, Path, Path]:
    base = ROOT / "figures_cross_cell_20260718"
    return (
        base / "panel_a_overall_mean_metrics" / "figure4a_cross_cell_overall_mean_metrics.svg",
        base / "panels_b_g_quantitative_mean_txpert" / "figure_cross_cell_coverage_b_e_mean_txpert.svg",
        base / "panels_f_g_hspa9" / "rendered_20260721" / "figure4f_hspa9_external_targets.svg",
        base / "panels_f_g_hspa9" / "csa_heatmap_20260721" / "figure4g_hspa9_csa_heatmap.svg",
    )


def make_split_figure4_a_to_e() -> None:
    """Create the cross-cell benchmark figure, reserving room for anchor diagnostics."""
    width, scale, shift, gap = 509.76, 0.92, 10.0, 5.67
    top_path, middle_path, _, _ = figure4_sources()
    _, top_height = append_dimensions(top_path, scale)
    _, middle_height = append_dimensions(middle_path, scale)
    total_height = top_height + gap + middle_height
    tree = master(width, total_height, "figure4_cross_cell_a_to_e_editable.svg")
    root = tree.getroot()
    graphics = etree.SubElement(root, q("g"), id="graphics")
    graphics.set(iq("groupmode"), "layer")
    graphics.set(iq("label"), "Vector panels")
    append_panel(graphics, top_path, panel_id="fig4_a", x=shift, y=0, scale=scale)
    append_panel(graphics, middle_path, panel_id="fig4_b_to_e", x=shift, y=top_height + gap, scale=scale)
    tree.write(
        str(OUT / "figure4_cross_cell_a_to_e_editable.svg"),
        encoding="utf-8", xml_declaration=True, pretty_print=True,
    )


def make_split_figure4_a_to_f() -> None:
    """Append the control-anchor diagnostic to the cross-cell benchmark panels."""
    width, scale, shift, gap = 509.76, 0.92, 10.0, 5.67
    top_margin = 3.0 * 72.0 / 25.4
    top_path, middle_path, _, _ = figure4_sources()
    anchor_path = (
        ROOT
        / "figures_cross_cell_20260723"
        / "panel_control_context_swap_fig4_style_20260723"
        / "figure4f_control_context_swap.svg"
    )
    _, top_height = append_dimensions(top_path, scale)
    _, middle_height = append_dimensions(middle_path, scale)
    anchor_width, anchor_height = append_dimensions(anchor_path, 1.0)
    anchor_scale = width / anchor_width
    anchor_height *= anchor_scale
    anchor_y = top_margin + top_height + gap + middle_height + gap
    total_height = anchor_y + anchor_height

    tree = master(width, total_height, "figure4_cross_cell_a_to_f_editable.svg")
    root = tree.getroot()
    graphics = etree.SubElement(root, q("g"), id="graphics")
    graphics.set(iq("groupmode"), "layer")
    graphics.set(iq("label"), "Vector panels")
    append_panel(graphics, top_path, panel_id="fig4_a", x=shift, y=top_margin, scale=scale)
    append_panel(
        graphics,
        middle_path,
        panel_id="fig4_b_to_e",
        x=shift,
        y=top_margin + top_height + gap,
        scale=scale,
    )
    append_panel(
        graphics,
        anchor_path,
        panel_id="fig4_f_control_anchor",
        x=0,
        y=anchor_y,
        scale=anchor_scale,
    )
    tree.write(
        str(OUT / "figure4_cross_cell_a_to_f_editable.svg"),
        encoding="utf-8", xml_declaration=True, pretty_print=True,
    )


def make_split_figure5_f_to_g() -> None:
    """Create the HSPA9 case-study figure with target and CSA evidence."""
    width, scale, shift, gap = 509.76, 0.92, 10.0, 4.0
    top_margin = 3.0 * 72.0 / 25.4  # 3 mm breathing room above panel a.
    _, _, f_path, g_path = figure4_sources()
    _, f_height = append_dimensions(f_path, scale)
    _, g_height = append_dimensions(g_path, width / 510.264)
    total_height = top_margin + f_height + gap + g_height
    tree = master(width, total_height, "figure5_hspa9_f_to_g_editable.svg")
    root = tree.getroot()
    graphics = etree.SubElement(root, q("g"), id="graphics")
    graphics.set(iq("groupmode"), "layer")
    graphics.set(iq("label"), "Vector panels")
    append_panel(
        graphics, f_path, panel_id="fig5_a", x=shift, y=top_margin, scale=scale,
        panel_label_map={"f": "a"}, left_align_hspa_title=True,
    )
    append_panel(
        graphics, g_path, panel_id="fig5_b", x=-25, y=top_margin + f_height + gap, scale=width / 510.264,
        panel_label_map={"g": "b"},
    )
    tree.write(
        str(OUT / "figure5_hspa9_f_to_g_editable.svg"),
        encoding="utf-8", xml_declaration=True, pretty_print=True,
    )


def append_dimensions(path: Path, scale: float) -> tuple[float, float]:
    root = etree.parse(str(path)).getroot()
    _, _, width, height = parse_viewbox(root)
    return width * scale, height * scale


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure2()
    make_figure4()
    make_split_figure4_a_to_e()
    make_split_figure4_a_to_f()
    make_split_figure5_f_to_g()
    (OUT / "README.md").write_text(
        "# Editable SVG masters\n\n"
        "These masters compose native vector SVG panels directly and contain no external image links. "
        "All labels remain editable SVG text. Normal labels use Arial 6 pt regular black; panel "
        "letters use Arial 8 pt bold black. The split Figure 4 is available both as a-e and a-f "
        "with the control-anchor diagnostic, while Figure 5 contains the HSPA9 case study (f-g). Export the "
        "opened SVG from Inkscape as PDF for manuscript use.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
