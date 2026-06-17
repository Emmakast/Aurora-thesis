#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/results/physics_aurora_ao81_steered.csv")
    parser.add_argument("--output", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/results/physics_alpha_table.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # We want mean model_value per alpha and metric
    # But wait, what if we want difference from era5?
    # Let's just use the raw mean model_value for now, or relative error.
    # Actually, for RMSE, we want the RMSE.
    
    # Let's group by metric and alpha and get mean
    # We drop nans
    agg = df.groupby(["metric_name", "alpha"])["model_value"].mean().reset_index()
    metrics_order = [
        "dry_mass_drift_pct_per_day",
        "water_mass_drift_pct_per_day",
        "total_energy_drift_pct_per_day",
        "spectrum_divergence",
        "spectrum_residual",
        "effective_resolution_km",
        "geostrophic_rmse_diff",
        "hydrostatic_rmse_diff",
        "lapse_rate_wasserstein"
    ]
    metric_labels = {
        "dry_mass_drift_pct_per_day": "Dry Mass Drift →0 [%/day]",
        "water_mass_drift_pct_per_day": "Water Mass Drift →0 [%/day]",
        "total_energy_drift_pct_per_day": "Total Energy Drift →0 [%/day]",
        "effective_resolution_km": "Eff. Resolution ↓111.5 [km]",
        "spectrum_residual": "Spec. Residual ↓0",
        "spectrum_divergence": "Spec. Divergence ↓0",
        "geostrophic_rmse_diff": "Geostrophic RMSE Δ →0 [m/s]",
        "hydrostatic_rmse_diff": "Hydrostatic RMSE Δ →0 [m²/s²]",
        "lapse_rate_wasserstein": "Mean Lapse Rate W-Dist ↓0"
    }
    metrics = [m for m in metrics_order if m in df["metric_name"].values]
    alphas = sorted(df["alpha"].dropna().unique())

    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    cell_texts = [["Metric"] + [f"Alpha = {a}" for a in alphas]]
    cell_colors = [[header_color] * len(cell_texts[0])]

    for metric in metrics:
        sub = agg[agg["metric_name"] == metric]
        if sub.empty:
            continue
            
        row_t = [metric_labels.get(metric, metric)]
        row_c = [white.copy()]

        vals = []
        # get all values to find max_abs for coloring
        for a in alphas:
            val = sub[sub["alpha"] == a]["model_value"]
            if not val.empty:
                vals.append(float(val.iloc[0]))
        
        # We need a baseline or way to define "worse". Usually higher is worse (redder).
        # We can scale intensity from min(vals) to max(vals) so min is white and max is red.
        # But for drift or diff, the absolute value is the error magnitude.
        
            
        vals = []
        for a in alphas:
            val = sub[sub["alpha"] == a]["model_value"]
            if not val.empty:
                vals.append(float(val.iloc[0]))
                
        if "diff" in metric or "drift" in metric:
            vals_for_color = [abs(v) for v in vals]
        else:
            vals_for_color = vals

        if len(vals_for_color) > 0:
            min_v, max_v = min(vals_for_color), max(vals_for_color)
            diff = max_v - min_v if max_v != min_v else 1.0
        else:
            diff = 1.0
            min_v = 0.0

        for a in alphas:
            val = sub[sub["alpha"] == a]["model_value"]
            if val.empty:
                row_t.append("—")
                row_c.append(white.copy())
            else:
                v = float(val.iloc[0])
                if "rmse" in metric or "resolution" in metric:
                    row_t.append(f"{v:.2f}")
                elif "mass_Eg" in metric or "mass_kg" in metric or "energy_J" in metric:
                    row_t.append(f"{v:.4e}")
                elif "drift" in metric:
                    row_t.append(f"{v:+.3f}%")
                elif "wasserstein" in metric:
                    row_t.append(f"{v:.3f}")
                else:
                    row_t.append(f"{v:.4f}")
                
                color_val = abs(v) if ("diff" in metric or "drift" in metric) else v
                intensity = (color_val - min_v) / diff
                intensity *= 0.8 # max 80% red
                row_c.append(white * (1 - intensity) + red * intensity)

        cell_texts.append(row_t)
        cell_colors.append(row_c)

    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)

    fig, ax = plt.subplots(figsize=(max(2.0 * n_cols, 10), max(0.5 * n_rows, 3.0)))
    ax.axis("off")

    colWidths = [0.45] + [0.15] * len(alphas)
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=colWidths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.6)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor(tuple(header_color))
        
    for r in range(1, n_rows):
        table[r, 0].set_text_props(fontweight="bold")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Table saved to {args.output}")

if __name__ == "__main__":
    main()
