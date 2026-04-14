#!/usr/bin/env python3
"""
Plot relative conservation metric timeseries for the 3-level ablation study.

Produces three separate figures (one per metric: dry mass, water mass, total energy),
each showing Aurora, Pangu, and HRES on the same axes (3-level ablation), plus
full-level Pangu and HRES as dashed lines for comparison.

Values are expressed as relative change from the first forecast step (t = 6 h)
per initialisation date, then averaged across all dates with ±1 std shading.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── configuration ──────────────────────────────────────────────────────────
RESULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
ABLATION_DIR = RESULT_DIR / "ablation_3levels"

# 3-level ablation models (solid lines)
ABLATION_MODELS = {
    "aurora": {"label": "Aurora (3 levels)", "color": "#e74c3c"},
    "pangu":  {"label": "Pangu (3 levels)",  "color": "#2980b9"},
    "hres":   {"label": "HRES (3 levels)",   "color": "#27ae60"},
}

# Full-level models (dashed lines)
FULL_MODELS = {
    "pangu": {"label": "Pangu (all levels)", "color": "#2980b9",
              "file": RESULT_DIR / "physics_timeseries_pangu_2022.csv"},
    "hres":  {"label": "HRES (all levels)",  "color": "#27ae60",
              "file": RESULT_DIR / "physics_timeseries_hres_2020.csv"},
}

METRICS = {
    "dry_mass_Eg":    {"label": "Dry Air Mass",  "unit": "%"},
    "water_mass_kg":  {"label": "Water Mass",    "unit": "%"},
    "total_energy_J": {"label": "Total Energy",  "unit": "%"},
}


def load_timeseries(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compute_relative(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return relative change (%) from t=6 h for each init date."""
    piv = df.pivot_table(index="date", columns="forecast_hour", values=col,
                         aggfunc="first")
    first = piv.iloc[:, 0]  # 6 h column
    rel = piv.subtract(first, axis=0).divide(first.abs(), axis=0) * 100.0
    return rel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default=str(ABLATION_DIR),
                        help="Directory to save plots")
    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load ablation data
    ablation_data = {}
    for m in ABLATION_MODELS:
        p = ABLATION_DIR / f"physics_timeseries_{m}_2022_3levels.csv"
        ablation_data[m] = load_timeseries(p)

    # load full-level data
    full_data = {}
    for m, info in FULL_MODELS.items():
        full_data[m] = load_timeseries(info["file"])

    # collect ablation forecast hours for filtering full-level data
    ablation_hours = set()
    for df in ablation_data.values():
        ablation_hours.update(df["forecast_hour"].unique())

    for metric_col, metric_info in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        # --- full-level models (dashed, plotted first so they sit behind) ---
        for model_key, model_info in FULL_MODELS.items():
            rel = compute_relative(full_data[model_key], metric_col)
            # keep only forecast hours that appear in the ablation runs
            common_hours = sorted(set(rel.columns) & ablation_hours)
            rel = rel[common_hours]
            hours = rel.columns.values
            mean = rel.mean(axis=0).values
            std  = rel.std(axis=0).values

            ax.plot(hours, mean, color=model_info["color"],
                    linewidth=1.8, linestyle="--", alpha=0.7,
                    label=model_info["label"])
            ax.fill_between(hours, mean - std, mean + std,
                            color=model_info["color"], alpha=0.07)

        # --- 3-level ablation models (solid) ---
        for model_key, model_info in ABLATION_MODELS.items():
            rel = compute_relative(ablation_data[model_key], metric_col)
            hours = rel.columns.values
            mean = rel.mean(axis=0).values
            std  = rel.std(axis=0).values

            ax.plot(hours, mean, color=model_info["color"],
                    linewidth=2.2, label=model_info["label"])
            ax.fill_between(hours, mean - std, mean + std,
                            color=model_info["color"], alpha=0.15)

        ax.set_xlabel("Lead time (hours)", fontsize=12)
        ax.set_ylabel(f"Relative change ({metric_info['unit']})", fontsize=12)
        ax.set_title(f"Level Ablation — {metric_info['label']}",
                     fontsize=13, fontweight="bold")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=9.5, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        fname = outdir / f"ablation_3levels_{metric_col.split('_')[0]}_mass.png"
        if "energy" in metric_col:
            fname = outdir / "ablation_3levels_total_energy.png"
        fig.savefig(fname, dpi=200)
        print(f"  ✓ Saved {fname}")
        plt.close(fig)

    print("\nDone — all 3 ablation plots saved.")


if __name__ == "__main__":
    main()
