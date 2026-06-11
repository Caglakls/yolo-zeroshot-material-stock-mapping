"""Render a YOLO-comparison overview figure from a full-results CSV.

Produces a single PNG with three panels:
  1. Dominant-material bar chart (count per canonical label).
  2. WWR histogram (20 bins over [0, 1]).
  3. Average material percentage bar chart (mean of each *_Pct column).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def _pct_column(label: str) -> str:
    return label.replace(" ", "_") + "_Pct"


def render_overview(csv_path: str, output_path: str, title: str):
    df = pd.read_csv(csv_path)

    labels = config.MATERIAL_LABELS
    pct_cols = [_pct_column(label) for label in labels]

    dom_counts = df["Dominant_Material"].value_counts()
    dom_values = [int(dom_counts.get(label, 0)) for label in labels]

    wwr_series = pd.to_numeric(df["WWR"], errors="coerce").dropna()
    wwr_series = wwr_series[(wwr_series >= 0.0) & (wwr_series <= 1.0)]

    avg_pct = [float(df[col].mean()) if col in df.columns else 0.0
               for col in pct_cols]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(labels, dom_values, color="#4C72B0")
    ax.set_title("Dominant Material Distribution")
    ax.set_ylabel("Number of Buildings")
    ax.tick_params(axis="x", rotation=30)
    for idx, v in enumerate(dom_values):
        ax.text(idx, v, str(v), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    if len(wwr_series) > 0:
        ax.hist(wwr_series, bins=20, range=(0.0, 1.0),
                color="#55A868", edgecolor="black")
    ax.set_title(f"WWR Distribution (n={len(wwr_series)})")
    ax.set_xlabel("Window-to-Wall Ratio")
    ax.set_ylabel("Number of Buildings")
    ax.set_xlim(0.0, 1.0)

    ax = axes[2]
    ax.bar(labels, avg_pct, color="#C44E52")
    ax.set_title("Average Material Percentage")
    ax.set_ylabel("Percent (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(100.0, max(avg_pct) * 1.1 if avg_pct else 100.0))
    for idx, v in enumerate(avg_pct):
        ax.text(idx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overview to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render results_overview PNG from a full-results CSV")
    parser.add_argument("--input", required=True, help="Path to CSV")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", required=True, help="Figure title")
    args = parser.parse_args()
    render_overview(args.input, args.output, args.title)


if __name__ == "__main__":
    main()
