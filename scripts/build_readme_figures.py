"""Render the README's figures to assets/, once for GitHub's light page and once for dark.

    python3 scripts/build_readme_figures.py

Writes:
    assets/architecture-{light,dark}.svg   via build_architecture_diagram.build
    assets/embeddings-{light,dark}.svg     the embedding-model comparison

The README shows the matching file through <picture>, and keeps each figure's
numbers as a table beside it, so the text still carries everything when an image
does not load.

The embedding numbers are the measured ones recorded in
databricks/navigator_config.yaml (from scripts/compare_embedding_models.py).
Change them there and here together, or re-run that script first.

Colour: the chosen model is the one blue mark in each panel and the others are a
neutral grey. Identity is carried by the row labels, never by colour, so the grey
is de-emphasis, not a second series. Checked with the dataviz validator on both
surfaces: separation and contrast pass; its gray-chroma check flags the grey,
which is intended here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_architecture_diagram  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

MODELS = ["qwen3-embedding-0-6b", "bge-large-en", "gte-large-en"]
CHOSEN = 0
# (title, values, format, axis maximum). Each panel keeps its own scale: the
# four measures are in different units, so they are small multiples, never one axis.
PANELS = [
    ("Recall@5", [0.94, 0.92, 0.92], "{:.2f}", 1.0),
    ("MRR", [0.860, 0.802, 0.770], "{:.2f}", 1.0),
    ("Score margin, correct vs incorrect", [0.279, 0.142, 0.161], "+{:.3f}", 0.32),
    ("Sustained chunks/s", [21.4, 20.9, 2.0], "{:.1f}", 24.0),
]

THEMES = {
    "light": {"surface": "#FFFFFF", "ink": "#1F2328", "muted": "#59636E",
              "grid": "#D1D9E0", "chosen": "#2A78D6", "other": "#8C959F"},
    "dark": {"surface": "#0D1117", "ink": "#E6EDF3", "muted": "#9198A1",
             "grid": "#30363D", "chosen": "#3987E5", "other": "#6E7681"},
}


def embeddings_chart(output: Path, theme_name: str) -> None:
    theme = THEMES[theme_name]
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), facecolor=theme["surface"],
                             gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.155, right=0.985, top=0.72, bottom=0.08)
    fig.text(0.012, 0.93, "Embedding model comparison", fontsize=13,
             fontweight="bold", color=theme["ink"], ha="left")
    fig.text(0.012, 0.845, "Same corpus and questions. The blue model is the one "
             "deployed; it leads every measure.", fontsize=9.5, color=theme["muted"],
             ha="left")

    rows = list(range(len(MODELS)))[::-1]  # First model on top.
    for index, (ax, (title, values, fmt, limit)) in enumerate(zip(axes, PANELS)):
        ax.set_facecolor(theme["surface"])
        colours = [theme["chosen"] if i == CHOSEN else theme["other"]
                   for i in range(len(MODELS))]
        ax.barh(rows, values, height=0.56, color=colours, edgecolor=theme["surface"],
                linewidth=2)
        for row, value in zip(rows, values):
            ax.text(value + limit * 0.02, row, fmt.format(value), va="center",
                    ha="left", fontsize=9, color=theme["ink"])
        ax.set_xlim(0, limit * 1.22)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.set_title(title, fontsize=9.5, color=theme["ink"], loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks(rows)
        ax.set_yticklabels(MODELS if index == 0 else [""] * len(MODELS),
                           fontsize=9, color=theme["ink"])
        ax.tick_params(axis="y", length=0, pad=6)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(theme["grid"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=theme["surface"])
    plt.close(fig)


def main() -> int:
    for theme in ("light", "dark"):
        architecture = ASSETS / f"architecture-{theme}.svg"
        build_architecture_diagram.build(architecture, theme)
        embeddings = ASSETS / f"embeddings-{theme}.svg"
        embeddings_chart(embeddings, theme)
        for path in (architecture, embeddings):
            print(f"  wrote {path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
