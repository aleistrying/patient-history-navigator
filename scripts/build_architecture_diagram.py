"""Render Figure 1, the end-to-end architecture diagram, to a PNG.

The figure used to exist only as `word/media/image1.png` inside the study guide, so
correcting it meant editing a binary. It is drawn here instead, and
`scripts/refresh_study_guide.py` swaps the rendered bytes into the .docx, so the
figure can be corrected again the same way any other text can.

    python3 scripts/build_architecture_diagram.py [output.png]

Defaults to `deliverables/architecture_diagram.png`.

The canvas is 1700x900 pixels because the .docx frame declares 7.00 x 3.71 inches
for this picture and Word stretches the bytes to fill it. Matching the original
pixel aspect is what keeps the replacement from being distorted; change `SIZE` only
alongside the frame.

Every label is a name that appears in `databricks/navigator_config.yaml` or in the
module it describes. matplotlib is a documentation-build dependency only -- nothing
in `databricks/requirements.txt` needs it, and the served model never imports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display on a build machine, and none is needed.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "deliverables" / "architecture_diagram.png"

# Pixel canvas. figsize x dpi = exactly these numbers, so a coordinate below is a
# pixel and the crop is predictable.
WIDTH, HEIGHT, DPI = 1700, 900, 200

INK = "#16223C"          # Titles and borders' darkest value.
DETAIL = "#41506B"       # Body lines: lighter than the title, still print-safe.
FOOTNOTE = "#5A6579"
ARROW = "#16223C"

# One fill per stage, as the original figure had. The fills are pale enough that
# they all reduce to a light grey in a greyscale print, so the border weight and the
# text carry the structure rather than the colour.
BOXES_TOP = [
    ("Synthetic sources", ["Synthea FHIR JSON", "PDF / Markdown / text"],
     "#E7EDFA", "#2F5DA8"),
    ("Extraction", ["pypdf first", "ai_parse_document fallback"],
     "#E3F3F0", "#1E7A6E"),
    ("Preparation", ["FHIR normalisation", "LangChain recursive chunks"],
     "#EEE8FA", "#6A4CA8"),
    ("Indexing", ["Qwen3 embeddings", "Delta + AI Search"],
     "#FBF1DD", "#B4801B"),
    ("Serving", ["ResponsesAgent", "Model Serving"],
     "#FBE8E8", "#B03A3A"),
]

BOXES_BOTTOM = [
    ("Focused Python tools",
     ["8 tools: timeline, document comparison,",
      "lab trend, medication changes,",
      "count, list, read, patient selection"],
     "#EAF2E3", "#4E7A34"),
    ("Grounded RAG",
     ["Reformulation + decomposition",
      "+ multi-query, patient filter,",
      "hybrid search, RRF"],
     "#E7EDFA", "#2F5DA8"),
    ("Persistent state",
     ["Unity Catalog memory store",
      "Query/retrieval TTL cache",
      "MLflow traces and evaluation"],
     "#E0F0F0", "#1E7A7A"),
]

TITLE = "Patient History Navigator: end-to-end architecture"
FOOT = ("Every retrieval request is filtered by patient_id. "
        "Tools query the full-document table; RAG queries the chunk index.")

# Geometry. The top row is five boxes and the bottom row three, both inset by the
# same margin so the two rows line up at their outer edges. The original figure ran
# its fifth box off the right edge and let several labels spill past their borders;
# MARGIN is applied on both sides here, and `check_fit` below measures every label
# against the box that holds it, so neither can happen unnoticed.
MARGIN = 40
TOP_W, TOP_H, TOP_Y = 300, 132, 560
BOT_W, BOT_H, BOT_Y = 500, 158, 175
TOP_GAP = (WIDTH - 2 * MARGIN - 5 * TOP_W) / 4
BOT_GAP = (WIDTH - 2 * MARGIN - 3 * BOT_W) / 2
BUS_Y = 452  # The horizontal run of the fan-out from Serving.

# Point sizes, chosen so the longest label in each row clears its border. At 200 dpi
# a point is 2.78 pixels, and the .docx scales 1700 pixels down to 7 inches, so a
# 6.7 pt label here reads as about 7.5 pt on the printed page.
TITLE_SIZE, BOX_TITLE_SIZE, DETAIL_SIZE, FOOT_SIZE = 16, 9.2, 6.6, 6.9
PAD_X = 16       # Inset of text from the box's left border.
TITLE_DROP = 34  # Baseline of the box title below its top border.
LINE_STEP = 24   # Pixels between detail lines.


def draw_box(ax, x, y, w, h, title, lines, fill, edge) -> list:
    """One rounded box: bold dark title, smaller grey detail lines beneath it.

    Returns the text artists paired with the right-hand limit they must stay inside,
    for `check_fit` to measure once a renderer exists.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=12",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
        # Rounded corners are drawn inside the given rectangle, so the box occupies
        # exactly the space the geometry above reserved for it.
        mutation_aspect=1,
    ))
    limit = x + w - PAD_X
    placed = [ax.text(x + PAD_X, y + h - TITLE_DROP, title, fontsize=BOX_TITLE_SIZE,
                      fontweight="bold", color=INK, ha="left", va="center")]
    for i, line in enumerate(lines):
        placed.append(ax.text(
            x + PAD_X, y + h - TITLE_DROP - 28 - i * LINE_STEP, line,
            fontsize=DETAIL_SIZE, color=DETAIL, ha="left", va="center"))
    return [(artist, limit) for artist in placed]


def check_fit(fig, ax, measured: list) -> list[str]:
    """Report any label whose rendered width runs past the border meant to hold it.

    The point sizes above were chosen by measuring rather than by eye, and this is
    what keeps them honest: change a label and the overflow is printed rather than
    discovered in the printed guide.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflows = []
    for artist, limit in measured:
        box = artist.get_window_extent(renderer)
        right = ax.transData.inverted().transform((box.x1, box.y1))[0]
        if right > limit:
            overflows.append(f"{artist.get_text()!r} overruns by "
                             f"{right - limit:.0f} px")
    return overflows


def arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.6, color=ARROW, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="butt",
    ))


def build(output: Path) -> Path:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.set_facecolor("white")

    measured = [(ax.text(MARGIN, HEIGHT - 62, TITLE, fontsize=TITLE_SIZE,
                         fontweight="bold", color=INK, ha="left", va="center"),
                 WIDTH - MARGIN)]

    top_x = []
    for i, (title, lines, fill, edge) in enumerate(BOXES_TOP):
        x = MARGIN + i * (TOP_W + TOP_GAP)
        top_x.append(x)
        measured += draw_box(ax, x, TOP_Y, TOP_W, TOP_H, title, lines, fill, edge)

    # Left to right through the offline pipeline into serving.
    for i in range(4):
        start = top_x[i] + TOP_W
        arrow(ax, (start + 4, TOP_Y + TOP_H / 2), (top_x[i + 1] - 4, TOP_Y + TOP_H / 2))

    bot_centres = []
    for i, (title, lines, fill, edge) in enumerate(BOXES_BOTTOM):
        x = MARGIN + i * (BOT_W + BOT_GAP)
        bot_centres.append(x + BOT_W / 2)
        measured += draw_box(ax, x, BOT_Y, BOT_W, BOT_H, title, lines, fill, edge)

    # Serving fans out to all three: one drop, one horizontal bus, three arrowheads.
    serving_centre = top_x[4] + TOP_W / 2
    ax.plot([serving_centre, serving_centre], [TOP_Y, BUS_Y],
            color=ARROW, linewidth=1.6, solid_capstyle="butt")
    ax.plot([min(bot_centres), serving_centre], [BUS_Y, BUS_Y],
            color=ARROW, linewidth=1.6, solid_capstyle="butt")
    for centre in bot_centres:
        arrow(ax, (centre, BUS_Y), (centre, BOT_Y + BOT_H + 4))

    measured.append((ax.text(MARGIN, 52, FOOT, fontsize=FOOT_SIZE, color=FOOTNOTE,
                             ha="left", va="center"), WIDTH - MARGIN))

    for note in check_fit(fig, ax, measured):
        print(f"  WARNING: {note}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)
    return output


def main(argv: list[str]) -> int:
    output = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUTPUT
    build(output)
    from PIL import Image  # Only to report what was written.

    with Image.open(output) as image:
        try:
            shown = output.relative_to(ROOT)
        except ValueError:  # An output written outside the repository, e.g. a scratch check.
            shown = output
        print(f"  wrote {shown}  {image.size[0]}x{image.size[1]} at {DPI} dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
