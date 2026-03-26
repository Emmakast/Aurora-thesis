#!/usr/bin/env python3
"""
Plot Pangu-Weather 2020 vs 2022 timeseries side by side.
For conservation metrics: relative change from t=6h.
For balance metrics: absolute values.
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

FILES = {
    "Pangu 2020": RESULT_DIR / "physics_timeseries_pangu_2020.csv",
    "Pangu 2022": RESULT_DIR / "physics_timeseries_pangu_2022.csv",
}

COLORS = {
    "Pangu 2020": "#2980b9",
    "Pangu 2022": "#e74c3c",
}

CONSERVATION = {"dry_mass_Eg", "water_mass_kg", "total_energy_J"}

METRICS = {
    "dry_mass_Eg":       "Dry Air Mass",
    "water_mass_kg":     "Water Mass",
    "total_energy_J":    "Total Energy",
    "hydrostatic_rmse":  "Hydrostatic RMSE",
    "geostrophic_rmse":  "Geostrophic RMSE",
}


def compute_relative(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot_table(index="date", columns="forecast_hour", values=col, aggfunc="first")
    first = piv.iloc[:, 0]
    return piv.subtract(first, axis=0).divide(first.abs(), axis=0) * 100.0


def main():
    outdir = RESULT_DIR / "plots_pangu_comparison"
    outdir.mkdir(exist_ok=True)

    data = {}
    for label, path in FILES.items():
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.drop_duplicates(subset=["date", "forecast_hour"])
        data[label] = df

    for col, title in METRICS.items():
        if col in CONSERVATION:
            # ── Relative plot ──
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for label, df in data.items():
                if col not in df.columns:
                    continue
                rel = compute_relative(df, col)
                hours = rel.columns.values
                mean = rel.mean(axis=0).values
                std  = rel.std(axis=0).values
                ax.plot(hours, mean, color=COLORS[label], linewidth=2.2,
                        marker="o", markersize=5, label=label)
                ax.fill_between(hours, mean - std, mean + std,
                                color=COLORS[label], alpha=0.15)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Relative change (%)", fontsize=12)
            ax.set_title(f"{title} — Relative Change from t₀", fontsize=13, fontweight="bold")
        else:
            # ── Absolute plot ──
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for label, df in data.items():
                if col not in df.columns:
                    continue
                agg = df.groupby("forecast_hour")[col].agg(["mean", "std"]).reset_index()
                ax.plot(agg["forecast_hour"], agg["mean"], color=COLORS[label],
                        linewidth=2.2, marker="o", markersize=5, label=label)
                ax.fill_between(agg["forecast_hour"],
                                agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"],
                                color=COLORS[label], alpha=0.15)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"{title} over Forecast Horizon", fontsize=13, fontweight="bold")

        ax.set_xlabel("Lead time (hours)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fname = outdir / f"pangu_2020_vs_2022_{col}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"  ✓ {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
