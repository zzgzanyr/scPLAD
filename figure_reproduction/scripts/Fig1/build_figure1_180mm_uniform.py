#!/usr/bin/env python3
"""Build a 180 mm Figure 1 with one final-size font and aligned labels."""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "ppt_text_rebuild"
BASE_SVG = SOURCE_DIR / "figure1_8x8_no_text_singlepage.svg"
MANIFEST = SOURCE_DIR / "figure1_8x8_text_manifest.json"
OUTPUT_SVG = HERE / "figure1_180mm_arial6pt_regular_3mmgaps_equalmargins.svg"
OUTPUT_PDF = HERE / "figure1_180mm_arial6pt_regular_3mmgaps_equalmargins.pdf"
RAW_PDF = HERE / "figure1_180mm_arial6pt_regular_3mmgaps_equalmargins_inkscape_raw.pdf"
OUTPUT_PNG = HERE / "figure1_180mm_arial6pt_regular_3mmgaps_equalmargins_preview.png"
INKSCAPE = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")

PAGE_WIDTH_MM = 180.0
FONT_SIZE_PT = 6.0
PANEL_LABEL_SIZE_PT = 8.0
PANEL_GAP_MM = 3.0
OUTER_MARGIN_MM = 3.0
LINE_HEIGHT = 1.16
OUTER_FRAME_STROKE = 2.4

VIEW_WIDTH = 1598.2287
A_FRAME_BOTTOM = 958.0
B_FRAME_Y = 987.78937
B_FRAME_HEIGHT = 235.0
C_FRAME_X = 32.0
C_FRAME_Y = 1288.5677
C_FRAME_WIDTH = 380.7245
D_FRAME_X = 424.8710
D_FRAME_WIDTH = 1147.1290

PANEL_GAP_USER = PANEL_GAP_MM * VIEW_WIDTH / PAGE_WIDTH_MM
B_SHIFT_Y = A_FRAME_BOTTOM + PANEL_GAP_USER - B_FRAME_Y
B_FRAME_TARGET_Y = B_FRAME_Y + B_SHIFT_Y
B_FRAME_TARGET_BOTTOM = B_FRAME_TARGET_Y + B_FRAME_HEIGHT
CD_SHIFT_Y = B_FRAME_TARGET_BOTTOM + PANEL_GAP_USER - C_FRAME_Y
C_FRAME_TARGET_Y = C_FRAME_Y + CD_SHIFT_Y
D_SHIFT_X = C_FRAME_X + C_FRAME_WIDTH + PANEL_GAP_USER - D_FRAME_X
D_FRAME_TARGET_X = D_FRAME_X + D_SHIFT_X
D_FRAME_TARGET_WIDTH = D_FRAME_WIDTH - D_SHIFT_X
D_SCALE_X = D_FRAME_TARGET_WIDTH / D_FRAME_WIDTH
D_TRANSLATE_X = D_FRAME_TARGET_X - D_SCALE_X * D_FRAME_X

PANEL_B_LABEL_Y = (A_FRAME_BOTTOM + B_FRAME_TARGET_Y) / 2
PANEL_CD_LABEL_Y = (B_FRAME_TARGET_BOTTOM + C_FRAME_TARGET_Y) / 2

# Calibrated against the rendered PDF: this is the top of the visible "a" glyph.
TOP_VISIBLE_CONTENT_Y = 27.8
BOTTOM_VISIBLE_CONTENT_Y = C_FRAME_TARGET_Y + 513.7316 + OUTER_FRAME_STROKE / 2
OUTER_MARGIN_USER = OUTER_MARGIN_MM * VIEW_WIDTH / PAGE_WIDTH_MM
VIEWBOX_MIN_Y = TOP_VISIBLE_CONTENT_Y - OUTER_MARGIN_USER
VIEWBOX_HEIGHT = (
    BOTTOM_VISIBLE_CONTENT_Y + OUTER_MARGIN_USER - VIEWBOX_MIN_Y
)

SKIP_TEXT_IDS = {"text25", "text27", "text29", "text31"}
OUTER_FRAME_IDS = {
    "panel_a_outer_frame_white",
    "fig1bcd_rect11",
    "fig1bcd_rect188",
    "fig1bcd_rect254",
}

GENE_SUBSCRIPTS = {
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "ₖ": "k",
    "ₘ": "m",
    "ₚ": "p",
    "ₛ": "s",
    "ᵤ": "u",
}


# Values are SVG anchor coordinates, not bounding-box centres.
X_OVERRIDES = {
    "fig1a_text73": 73.0,
    "fig1a_text294": 73.0,
    "fig1a_text470": 73.0,
    "fig1a_text239": 288.0,
    "fig1a_text239-4": 288.0,
    "fig1a_text239-1": 288.0,
    "fig1a_text239-7": 500.0,
    "fig1a_text239-0": 500.0,
    "fig1a_text239-0-0": 500.0,
    "fig1a_text240": 282.0,
    "fig1a_text416": 282.0,
    "fig1a_text525": 282.0,
    "fig1a_text241": 348.0,
    "fig1a_text417": 348.0,
    "fig1a_text526": 348.0,
    "fig1a_text242": 408.0,
    "fig1a_text418": 408.0,
    "fig1a_text527": 408.0,
    "fig1a_text243": 462.0,
    "fig1a_text419": 462.0,
    "fig1a_text528": 462.0,
    "text25": 85.0,
    "text26": 85.0,
    "text27": 85.0,
    "text28": 85.0,
    "text29": 85.0,
    "text30": 85.0,
    "text31": 85.0,
    "text32": 85.0,
    "fig1bcd_text92": 562.3,
    "fig1bcd_text94": 562.3,
    "fig1bcd_text121": 1149.4,
    "fig1bcd_text123": 1149.4,
    "fig1bcd_text350": 689.3,
    "fig1bcd_text352": 689.3,
    "fig1bcd_text411": 1311.4,
    "fig1bcd_text413": 1311.4,
    "fig1bcd_text373": 736.0,
    "fig1bcd_text374": 736.0,
    "fig1bcd_text378": 892.4,
    "fig1bcd_text379": 892.4,
    "fig1bcd_text383": 1048.0,
    "fig1bcd_text384": 1048.0,
    "fig1bcd_text477": 1479.0,
    "fig1bcd_text478": 1479.0,
    "fig1bcd_text373-3": 310.1,
    "fig1bcd_text374-1": 310.1,
    "panel_label_d": D_FRAME_TARGET_X - 7.3,
}


Y_GROUPS = [
    (["fig1a_text71", "fig1a_text631"], 99.0),
    (["fig1a_text73", "fig1a_text239", "fig1a_text239-7"], 145.0),
    (["fig1a_text294", "fig1a_text239-4", "fig1a_text239-0"], 422.0),
    (["fig1a_text470", "fig1a_text239-1", "fig1a_text239-0-0"], 674.0),
    (["fig1a_text240", "fig1a_text241", "fig1a_text242", "fig1a_text243"], 205.5),
    (["fig1a_text416", "fig1a_text417", "fig1a_text418", "fig1a_text419"], 478.5),
    (["fig1a_text525", "fig1a_text526", "fig1a_text527", "fig1a_text528"], 733.0),
    (
        [
            "fig1a_text766",
            "fig1a_text767",
            "fig1a_text768",
            "fig1a_text769",
            "fig1a_text770",
            "fig1a_text771",
        ],
        450.0,
    ),
    (["fig1a_text777", "fig1a_text779"], 548.0),
    (["fig1bcd_text60", "fig1bcd_text95", "fig1bcd_text188"], 1090.0),
    (["fig1bcd_text92", "fig1bcd_text121"], 1127.5),
    (["fig1bcd_text94", "fig1bcd_text123"], 1147.5),
    (["fig1bcd_text350", "fig1bcd_text411"], 1587.5),
    (["fig1bcd_text352", "fig1bcd_text413"], 1607.5),
    (["fig1bcd_text373", "fig1bcd_text378", "fig1bcd_text383"], 1729.0),
    (["fig1bcd_text374", "fig1bcd_text379", "fig1bcd_text384"], 1749.5),
    (["panel_label_c", "panel_label_d"], 1259.0),
]

Y_OVERRIDES = {item_id: y for ids, y in Y_GROUPS for item_id in ids}
Y_OVERRIDES.update(
    {
        "fig1a_text593": 88.0,
        "fig1a_text594": 110.0,
        "text22": 1335.3,
        "text26": 1503.5,
        "text28": 1542.5,
        "text30": 1581.5,
        "text32": 1620.5,
        "panel_label_b": PANEL_B_LABEL_Y,
        "panel_label_c": PANEL_CD_LABEL_Y,
        "panel_label_d": PANEL_CD_LABEL_Y,
    }
)


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def styled_line(line: str, role: str) -> str:
    if role == "gene_symbol" and len(line) == 2 and line[0] == "g":
        subscript = GENE_SUBSCRIPTS.get(line[1])
        if subscript is not None:
            return (
                "<tspan>g</tspan>"
                '<tspan baseline-shift="sub">'
                f"{subscript}</tspan>"
            )
    return html.escape(line)


def anchor_position(record: dict) -> tuple[float, float]:
    bbox = record["bbox"]
    if record.get("anchor", "middle") == "start":
        x = float(bbox["left"])
    else:
        x = float(bbox["left"]) + float(bbox["width"]) / 2
    y = float(bbox["top"]) + float(bbox["height"]) / 2
    item_id = record["id"]
    x = X_OVERRIDES.get(item_id, x)
    y = Y_OVERRIDES.get(item_id, y)

    if item_id not in {"panel_label_a", "panel_label_b", "panel_label_c", "panel_label_d"}:
        top = float(bbox["top"])
        if 950.0 <= top < 1240.0:
            y += B_SHIFT_Y
        elif top >= 1240.0:
            y += CD_SHIFT_Y
            if float(bbox["left"]) >= 410.0:
                x = D_SCALE_X * x + D_TRANSLATE_X
    return x, y


def text_element(record: dict, font_size_user: float) -> str:
    anchor = record.get("anchor", "middle")
    text_anchor = "start" if anchor == "start" else "middle"
    x, center_y = anchor_position(record)
    lines = str(record["text"]).split("\n")
    is_panel_label = record["id"].startswith("panel_label_")
    effective_size = (
        font_size_user * PANEL_LABEL_SIZE_PT / FONT_SIZE_PT
        if is_panel_label
        else font_size_user
    )
    line_height = effective_size * LINE_HEIGHT
    first_y = center_y - (len(lines) - 1) * line_height / 2
    font_weight = "700" if is_panel_label else "400"
    font_style = "normal"
    element_id = html.escape(f"uniform_text_{record['id']}", quote=True)
    role = str(record.get("role", "body"))

    tspans = []
    for index, line in enumerate(lines):
        y = first_y + index * line_height
        tspans.append(
            f'    <tspan x="{fmt(x)}" y="{fmt(y)}">'
            f"{styled_line(line, role)}"
            "</tspan>"
        )

    return (
        f'  <text id="{element_id}" text-anchor="{text_anchor}" '
        f'font-family="Arial" font-size="{fmt(effective_size)}" '
        f'font-weight="{font_weight}" font-style="{font_style}" fill="#000000" '
        'dominant-baseline="middle" xml:space="preserve">\n'
        + "\n".join(tspans)
        + "\n  </text>"
    )


def update_page_size(svg: str, width_mm: float, height_mm: float) -> str:
    svg = re.sub(r'\bwidth="[^"]+"', f'width="{fmt(width_mm)}mm"', svg, count=1)
    svg = re.sub(r'\bheight="[^"]+"', f'height="{fmt(height_mm)}mm"', svg, count=1)
    svg = re.sub(
        r'\bviewBox="[^"]+"',
        f'viewBox="0 {fmt(VIEWBOX_MIN_Y)} {fmt(VIEW_WIDTH)} {fmt(VIEWBOX_HEIGHT)}"',
        svg,
        count=1,
    )

    page_match = re.search(r'<inkscape:page\b[^>]*/>', svg)
    if page_match is None:
        raise ValueError("Could not find Inkscape page definition")
    page_tag = page_match.group(0)
    for attribute, value in (
        ("x", 0.0),
        ("y", VIEWBOX_MIN_Y),
        ("width", VIEW_WIDTH),
        ("height", VIEWBOX_HEIGHT),
    ):
        page_tag = re.sub(
            rf'\b{attribute}="[^"]+"',
            f'{attribute}="{fmt(value)}"',
            page_tag,
            count=1,
        )
    svg = svg[: page_match.start()] + page_tag + svg[page_match.end() :]

    background_match = re.search(
        r'<rect\b(?=[^>]*\bid="figure_white_background")[^>]*/>', svg
    )
    if background_match is None:
        raise ValueError("Could not find figure background")
    background_tag = background_match.group(0)
    for attribute, value in (
        ("x", 0.0),
        ("y", VIEWBOX_MIN_Y),
        ("width", VIEW_WIDTH),
        ("height", VIEWBOX_HEIGHT),
    ):
        background_tag = re.sub(
            rf'\b{attribute}="[^"]+"',
            f'{attribute}="{fmt(value)}"',
            background_tag,
            count=1,
        )
    svg = (
        svg[: background_match.start()]
        + background_tag
        + svg[background_match.end() :]
    )
    return svg


def strengthen_outer_frames(svg: str) -> str:
    """Keep panel borders visible after PDF antialiasing and page scaling."""
    for element_id in OUTER_FRAME_IDS:
        pattern = (
            r'(<rect\b(?=[^>]*\bid="'
            + re.escape(element_id)
            + r'")[^>]*\bstroke-width=")[^"]+("[^>]*>)'
        )
        svg, count = re.subn(
            pattern,
            rf'\g<1>{fmt(OUTER_FRAME_STROKE)}\g<2>',
            svg,
            count=1,
        )
        if count != 1:
            raise ValueError(f"Could not update outer frame: {element_id}")
    return svg


def tag_start(svg: str, element_id: str) -> int:
    id_position = svg.find(f'id="{element_id}"')
    if id_position < 0:
        raise ValueError(f"Could not find SVG element: {element_id}")
    start = svg.rfind("<", 0, id_position)
    if start < 0:
        raise ValueError(f"Could not find start tag: {element_id}")
    return start


def reflow_panel_groups(svg: str) -> str:
    """Apply one 3 mm gutter to a-b, b-c/d, and c-d panel boundaries."""
    b_start = tag_start(svg, "fig1bcd_rect11")
    c_start = tag_start(svg, "fig1bcd_rect188")
    d_start = tag_start(svg, "fig1bcd_rect254")
    c_overlay_start = tag_start(svg, "figure1c_text_based_prior_sources")
    closing = svg.rfind("</svg>")
    if not (b_start < c_start < d_start < c_overlay_start < closing):
        raise ValueError("Unexpected panel ordering in source SVG")

    prefix = svg[:b_start]
    b_segment = svg[b_start:c_start]
    c_segment = svg[c_start:d_start]
    d_segment = svg[d_start:c_overlay_start]
    c_overlay_segment = svg[c_overlay_start:closing]
    return (
        prefix
        + f'<g id="panel_b_reflow" transform="translate(0,{fmt(B_SHIFT_Y)})">\n'
        + b_segment
        + "\n</g>\n"
        + f'<g id="panel_c_reflow" transform="translate(0,{fmt(CD_SHIFT_Y)})">\n'
        + c_segment
        + "\n</g>\n"
        + (
            '<g id="panel_d_reflow" '
            f'transform="matrix({fmt(D_SCALE_X)},0,0,1,{fmt(D_TRANSLATE_X)},{fmt(CD_SHIFT_Y)})">\n'
        )
        + d_segment
        + "\n</g>\n"
        + f'<g id="panel_c_compact_overlay_reflow" transform="translate(0,{fmt(CD_SHIFT_Y)})">\n'
        + c_overlay_segment
        + "\n</g>\n"
        + svg[closing:]
    )


def outer_frame_overlay() -> str:
    """Redraw panel frames above every white masking layer and panel object."""
    frames = (
        ("a", 32.0, 48.0, 1540.0, 910.0),
        ("b", 32.0, B_FRAME_TARGET_Y, 1540.0, B_FRAME_HEIGHT),
        ("c", C_FRAME_X, C_FRAME_TARGET_Y, C_FRAME_WIDTH, 513.7316),
        ("d", D_FRAME_TARGET_X, C_FRAME_TARGET_Y, D_FRAME_TARGET_WIDTH, 513.7316),
    )
    elements = [
        '<g id="outer_panel_frames_front" inkscape:groupmode="layer" '
        'inkscape:label="Outer panel frames - front">'
    ]
    for panel, x, y, width, height in frames:
        elements.append(
            f'  <rect id="outer_panel_{panel}_front" x="{fmt(x)}" y="{fmt(y)}" '
            f'width="{fmt(width)}" height="{fmt(height)}" rx="15" ry="15" '
            f'fill="none" stroke="#000000" stroke-width="{fmt(OUTER_FRAME_STROKE)}" '
            'stroke-linejoin="round"/>'
        )
    elements.append("</g>")
    return "\n".join(elements)


def build_svg() -> tuple[float, int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    view_width = float(manifest["slide_size"]["width"])
    page_height_mm = PAGE_WIDTH_MM * VIEWBOX_HEIGHT / view_width
    font_mm = FONT_SIZE_PT * 25.4 / 72.0
    font_size_user = font_mm * view_width / PAGE_WIDTH_MM

    svg = BASE_SVG.read_text(encoding="utf-8")
    svg = re.sub(
        r'\s*<g id="uniform_180mm_text_overlay"[^>]*>.*?</g>\s*',
        "\n",
        svg,
        flags=re.DOTALL,
    )
    svg = update_page_size(svg, PAGE_WIDTH_MM, page_height_mm)
    svg = strengthen_outer_frames(svg)
    svg = reflow_panel_groups(svg)

    overlay = [
        '<g id="uniform_180mm_text_overlay" '
        'inkscape:groupmode="layer" inkscape:label="Uniform Arial 6 pt labels">'
    ]
    visible_records = [
        record for record in manifest["records"] if record["id"] not in SKIP_TEXT_IDS
    ]
    overlay.extend(text_element(record, font_size_user) for record in visible_records)
    overlay.append("</g>")
    overlay.append(outer_frame_overlay())
    closing = svg.rfind("</svg>")
    if closing < 0:
        raise ValueError("No closing </svg> tag found")
    svg = svg[:closing] + "\n" + "\n".join(overlay) + "\n" + svg[closing:]
    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    return page_height_mm, len(visible_records)


def export() -> None:
    subprocess.run(
        [
            str(INKSCAPE),
            str(OUTPUT_SVG),
            "--export-area-page",
            f"--export-filename={RAW_PDF}",
        ],
        check=True,
    )
    normalize_pdf_page(RAW_PDF, OUTPUT_PDF)
    RAW_PDF.unlink(missing_ok=True)
    subprocess.run(
        [
            str(INKSCAPE),
            str(OUTPUT_SVG),
            "--export-area-page",
            "--export-dpi=180",
            f"--export-filename={OUTPUT_PNG}",
        ],
        check=True,
    )


def normalize_pdf_page(source: Path, output: Path) -> None:
    """Set an exact 180 mm MediaBox while preserving vector page content."""
    reader = PdfReader(source)
    source_page = reader.pages[0]
    source_width = float(source_page.mediabox.width)
    source_height = float(source_page.mediabox.height)

    target_width = PAGE_WIDTH_MM / 25.4 * 72.0
    target_height = target_width * VIEWBOX_HEIGHT / VIEW_WIDTH
    scale = min(target_width / source_width, target_height / source_height)
    translated_x = (target_width - source_width * scale) / 2.0
    translated_y = (target_height - source_height * scale) / 2.0

    page = PageObject.create_blank_page(width=target_width, height=target_height)
    transform = Transformation().scale(scale).translate(
        tx=translated_x, ty=translated_y
    )
    page.merge_transformed_page(source_page, transform)

    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    page_height_mm, label_count = build_svg()
    export()
    print(f"Page: {PAGE_WIDTH_MM:.1f} x {page_height_mm:.3f} mm")
    print(f"Uniform font: Arial {FONT_SIZE_PT:.1f} pt")
    print(f"Labels: {label_count}")
    print(f"SVG: {OUTPUT_SVG}")
    print(f"PDF: {OUTPUT_PDF}")
    print(f"Preview: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
