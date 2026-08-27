#!/usr/bin/env python3
"""Stack the finalized K562 benchmark and ablation figures vertically.

The source SVG/PDF files are left untouched. The composite uses a 180 mm
canvas and preserves vector content in SVG/PDF. Because both source canvases
already include vertical whitespace, their page boxes overlap by 3 mm to
produce an approximately 3 mm visible inter-panel gap.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import fitz
from lxml import etree
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TOP_SVG = ROOT / "figure3_180mm_6pt/figure3_k562_heldout_180mm_6pt.svg"
TOP_PDF = ROOT / "figure3_180mm_6pt/figure3_k562_heldout_180mm_6pt.pdf"
BOTTOM_SVG = (
    ROOT
    / "figures_k562_ablation_bar_20260716"
    / "figure4h_k562_ablation_bar_strip_20260716.svg"
)
BOTTOM_PDF = (
    ROOT
    / "figures_k562_ablation_bar_20260716"
    / "figure4h_k562_ablation_bar_strip_20260716.pdf"
)
OUT_DIR = ROOT / "figure3_with_k562_ablation_20260716"
STEM = "figure3_k562_heldout_with_ablation_180mm_20260716"

MM_TO_PT = 72.0 / 25.4
TARGET_WIDTH_MM = 180.0
PANEL_PAGE_OVERLAP_MM = 3.0


def _viewbox(root: etree._Element) -> tuple[float, float, float, float]:
    values = [float(value) for value in root.attrib["viewBox"].split()]
    if len(values) != 4:
        raise ValueError(f"Unexpected SVG viewBox: {root.attrib['viewBox']}")
    return tuple(values)  # type: ignore[return-value]


def _prefix_ids(element: etree._Element, prefix: str) -> None:
    """Avoid duplicate Matplotlib clip-path and marker IDs after nesting."""
    id_map: dict[str, str] = {}
    for node in element.iter():
        node_id = node.attrib.get("id")
        if node_id:
            new_id = f"{prefix}_{node_id}"
            id_map[node_id] = new_id
            node.attrib["id"] = new_id

    url_pattern = re.compile(r"url\(#([^)]+)\)")
    for node in element.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith("#") and value[1:] in id_map:
                node.attrib[key] = f"#{id_map[value[1:]]}"
                continue

            def replace_url(match: re.Match[str]) -> str:
                old = match.group(1)
                return f"url(#{id_map.get(old, old)})"

            node.attrib[key] = url_pattern.sub(replace_url, value)


def _nested_svg(source: Path, prefix: str, y: float, width: float) -> tuple[etree._Element, float]:
    source_root = etree.parse(str(source)).getroot()
    x0, y0, source_width, source_height = _viewbox(source_root)
    scale = width / source_width
    height = source_height * scale

    nested = etree.Element(
        "{http://www.w3.org/2000/svg}svg",
        x="0",
        y=f"{y:.6f}",
        width=f"{width:.6f}",
        height=f"{height:.6f}",
        viewBox=f"{x0:g} {y0:g} {source_width:g} {source_height:g}",
        preserveAspectRatio="xMidYMid meet",
    )
    for child in source_root:
        nested.append(copy.deepcopy(child))
    _prefix_ids(nested, prefix)
    return nested, height


def build_svg(output_path: Path) -> tuple[float, float]:
    target_width = TARGET_WIDTH_MM * MM_TO_PT
    gap = -PANEL_PAGE_OVERLAP_MM * MM_TO_PT

    top, top_height = _nested_svg(TOP_SVG, "benchmark", 0.0, target_width)
    bottom, bottom_height = _nested_svg(
        BOTTOM_SVG, "ablation", top_height + gap, target_width
    )
    target_height = top_height + gap + bottom_height

    nsmap = {None: "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
    root = etree.Element(
        "{http://www.w3.org/2000/svg}svg",
        nsmap=nsmap,
        width=f"{TARGET_WIDTH_MM:g}mm",
        height=f"{target_height / MM_TO_PT:.3f}mm",
        viewBox=f"0 0 {target_width:.6f} {target_height:.6f}",
        version="1.1",
    )
    root.append(top)
    root.append(bottom)
    etree.ElementTree(root).write(
        str(output_path), encoding="utf-8", xml_declaration=True, pretty_print=True
    )
    return target_width, target_height


def build_pdf(output_path: Path, target_width: float, target_height: float) -> None:
    top_doc = fitz.open(TOP_PDF)
    bottom_doc = fitz.open(BOTTOM_PDF)
    output = fitz.open()
    page = output.new_page(width=target_width, height=target_height)

    gap = -PANEL_PAGE_OVERLAP_MM * MM_TO_PT
    top_ratio = target_width / top_doc[0].rect.width
    top_height = top_doc[0].rect.height * top_ratio
    bottom_ratio = target_width / bottom_doc[0].rect.width
    bottom_height = bottom_doc[0].rect.height * bottom_ratio

    page.show_pdf_page(
        fitz.Rect(0, 0, target_width, top_height), top_doc, 0, keep_proportion=True
    )
    page.show_pdf_page(
        fitz.Rect(0, top_height + gap, target_width, top_height + gap + bottom_height),
        bottom_doc,
        0,
        keep_proportion=True,
    )
    output.set_metadata(
        {
            "title": "K562 held-out benchmark and component ablation",
            "subject": "Vector composite of finalized benchmark and ablation panels",
            "creator": "Python / PyMuPDF",
        }
    )
    output.save(output_path, garbage=4, deflate=True)
    output.close()
    top_doc.close()
    bottom_doc.close()


def render_pngs(pdf_path: Path, preview_path: Path, png600_path: Path) -> None:
    doc = fitz.open(pdf_path)
    page = doc[0]
    preview = page.get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72), alpha=False)
    preview.save(preview_path)
    publication = page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
    publication.save(png600_path)
    doc.close()

    with Image.open(png600_path) as image:
        image.save(png600_path, dpi=(600, 600), optimize=True)


def main() -> None:
    for source in (TOP_SVG, TOP_PDF, BOTTOM_SVG, BOTTOM_PDF):
        if not source.exists():
            raise FileNotFoundError(source)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / f"{STEM}.svg"
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    preview_path = OUT_DIR / f"{STEM}_preview.png"
    png600_path = OUT_DIR / f"{STEM}_600dpi.png"

    target_width, target_height = build_svg(svg_path)
    build_pdf(pdf_path, target_width, target_height)
    render_pngs(pdf_path, preview_path, png600_path)

    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    print(f"Preview: {preview_path}")
    print(f"PNG 600 dpi: {png600_path}")
    print(
        f"Canvas: {target_width / MM_TO_PT:.1f} x "
        f"{target_height / MM_TO_PT:.1f} mm"
    )


if __name__ == "__main__":
    main()
