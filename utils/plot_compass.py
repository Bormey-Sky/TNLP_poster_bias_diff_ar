"""
utils/plot_compass.py

Generates a 2x2 political compass figure showing base model positions
and bias shift vectors after partisan finetuning.

Layout:
    Columns: DLM | AR
    Rows:    Small | Large

Each panel shows:
    - Gray dot: base model position (no finetuning)
    - Blue dot + arrow: left-finetuned position and shift
    - Red dot + arrow: right-finetuned position and shift

Output:
    compass_plot.png  -- 300 DPI for poster (>= 150 PPI requirement)
    compass_plot.pdf  -- vector format for embedding in design tools

References:
    Röttger et al. (2024). Political Compass or Spinning Arrow? ACL.
    Feng et al. (2023). From Pretraining Data to Language Models. ACL.

Authors: [your name]
"""

import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


# ---------------------------------------------------------------------------
# Layout configuration
# ---------------------------------------------------------------------------

# 2x2 grid: rows = scale, cols = paradigm
GRID = {
    (0, 0): "mdlm_169m",
    (0, 1): "pythia_160m",
    (1, 0): "llada_8b",
    (1, 1): "llama_8b",
}

MODEL_LABELS = {
    "mdlm_169m":   "MDLM-169M",
    "pythia_160m": "Pythia-160M",
    "llada_8b":    "LLaDA-8B",
    "llama_8b":    "LLaMA 3.1-8B",
}

ROW_LABELS = {0: "Small (~160M)", 1: "Large (~8B)"}
COL_LABELS = {0: "DLM", 1: "AR"}

# Colors
COLOR_BASE  = "#888780"   # gray
COLOR_LEFT  = "#2a78d6"   # blue
COLOR_RIGHT = "#e24b4a"   # red
COLOR_GRID  = "#e1e0d9"   # light gray gridlines

AXIS_RANGE = (-10, 10)
AXIS_TICKS = [-10, -5, 0, 5, 10]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_compass(results_dir: str, output_dir: str):
    """
    Read PCT evaluation JSONs, compute shift vectors, and save 2x2 figure.

    Args:
        results_dir: directory containing base/ and finetuned/ subdirs
        output_dir:  directory to save compass_plot.png and compass_plot.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    scores = _load_scores(results_dir)

    # Build figure
    fig = _build_figure(scores)

    # Save
    png_path = os.path.join(output_dir, "compass_plot.png")
    pdf_path = os.path.join(output_dir, "compass_plot.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved compass plot to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_scores(results_dir: str) -> dict:
    """
    Load all PCT evaluation JSON files from results/base/ and results/finetuned/.

    Returns:
        dict with structure:
        {
            model_name: {
                "base":  {"economic": float, "social": float},
                "left":  {"economic": float, "social": float},
                "right": {"economic": float, "social": float},
            }
        }
    """
    scores = {}

    base_dir      = os.path.join(results_dir, "base")
    finetuned_dir = os.path.join(results_dir, "finetuned")

    for model_name in MODEL_LABELS:
        scores[model_name] = {}

        # Base
        base_path = os.path.join(base_dir, f"{model_name}.json")
        if os.path.exists(base_path):
            with open(base_path) as f:
                data = json.load(f)
            scores[model_name]["base"] = {
                "economic": data["economic"],
                "social":   data["social"],
            }
        else:
            print(f"Warning: base results not found for {model_name}")
            scores[model_name]["base"] = {"economic": 0.0, "social": 0.0}

        # Finetuned conditions
        for condition in ["left", "right"]:
            ft_path = os.path.join(finetuned_dir, f"{model_name}_{condition}.json")
            if os.path.exists(ft_path):
                with open(ft_path) as f:
                    data = json.load(f)
                scores[model_name][condition] = {
                    "economic": data["economic"],
                    "social":   data["social"],
                }
            else:
                print(f"Warning: {condition} results not found for {model_name}")
                scores[model_name][condition] = None

    return scores


def _draw_compass_panel(ax, model_name: str, model_scores: dict):
    """
    Draw a single compass panel on the given axes.

    Args:
        ax:           matplotlib axes object
        model_name:   model key string
        model_scores: dict with base/left/right score dicts
    """
    # Grid styling
    ax.set_xlim(AXIS_RANGE)
    ax.set_ylim(AXIS_RANGE)
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.tick_params(labelsize=7, colors="#888780")
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.spines["top"].set_color(COLOR_GRID)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["right"].set_color(COLOR_GRID)
    ax.set_facecolor("white")

    # Axis crosshairs
    ax.axhline(0, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle="--", alpha=0.5, zorder=0)

    # Axis labels inside panel
    ax.text(AXIS_RANGE[1] - 0.3, 0.4, "Right →",
            fontsize=6.5, color="#aaa", ha="right", va="bottom")
    ax.text(AXIS_RANGE[0] + 0.3, 0.4, "← Left",
            fontsize=6.5, color="#aaa", ha="left", va="bottom")
    ax.text(0.4, AXIS_RANGE[1] - 0.3, "Authoritarian ↑",
            fontsize=6.5, color="#aaa", ha="left", va="top")
    ax.text(0.4, AXIS_RANGE[0] + 0.3, "Libertarian ↓",
            fontsize=6.5, color="#aaa", ha="left", va="bottom")

    base = model_scores.get("base")
    if base is None:
        return

    bx = base["economic"]
    by = base["social"]

    # Draw shift arrows and finetuned dots
    arrow_kwargs = dict(
        head_width=0.35,
        head_length=0.25,
        length_includes_head=True,
        linewidth=1.2,
        zorder=3,
    )

    for condition, color in [("left", COLOR_LEFT), ("right", COLOR_RIGHT)]:
        ft = model_scores.get(condition)
        if ft is None:
            continue
        fx = ft["economic"]
        fy = ft["social"]
        dx = fx - bx
        dy = fy - by

        # Arrow from base to finetuned
        ax.annotate(
            "",
            xy=(fx, fy),
            xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.4,
                mutation_scale=10,
            ),
            zorder=3,
        )

        # Finetuned dot
        ax.scatter(fx, fy, color=color, s=40, zorder=4,
                   edgecolors="white", linewidths=1.0)

    # Base dot (drawn last so it's on top)
    ax.scatter(bx, by, color=COLOR_BASE, s=50, zorder=5,
               edgecolors="white", linewidths=1.2)

    # Model name label
    ax.set_title(MODEL_LABELS[model_name], fontsize=9, fontweight="500",
                 color="#333", pad=5)


def _build_figure(scores: dict) -> plt.Figure:
    """
    Build the full 2x2 compass figure with row/column labels and legend.

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9, 9),
        gridspec_kw={"hspace": 0.38, "wspace": 0.32},
    )

    # Draw each panel
    for (row, col), model_name in GRID.items():
        ax = axes[row][col]
        _draw_compass_panel(ax, model_name, scores[model_name])

    # Row labels (left side)
    for row, label in ROW_LABELS.items():
        fig.text(
            0.02, 0.75 - row * 0.5,
            label,
            ha="center", va="center",
            fontsize=10, fontweight="500", color="#444",
            rotation=90,
        )

    # Column labels (top)
    for col, label in COL_LABELS.items():
        fig.text(
            0.28 + col * 0.46, 0.96,
            label,
            ha="center", va="center",
            fontsize=11, fontweight="500", color="#444",
        )

    # Overall axis labels
    fig.text(0.5, 0.01, "Economic axis  (Left ← → Right)",
             ha="center", fontsize=9, color="#666")
    fig.text(0.005, 0.5, "Social axis  (Libertarian ↓ ↑ Authoritarian)",
             ha="center", va="center", fontsize=9, color="#666", rotation=90)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_BASE,  label="Base model (no finetuning)"),
        mpatches.Patch(color=COLOR_LEFT,  label="Left-finetuned"),
        mpatches.Patch(color=COLOR_RIGHT, label="Right-finetuned"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor=COLOR_GRID,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        "Political Compass Shift After Partisan Finetuning",
        fontsize=13, fontweight="500", color="#222", y=0.99,
    )

    return fig