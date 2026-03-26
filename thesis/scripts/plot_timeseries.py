"""
Timeseries plots for physics evaluation metrics.

Modes
-----
  single   : One model, individual metric plots with optional ERA5 baseline.
  combined : Multi-model overlay, auto-discovers all timeseries CSVs.

Usage
-----
  # Single model
  python plot_timeseries.py single results/physics_timeseries_pangu_2020.csv
  python plot_timeseries.py single results/physics_timeseries_pangu_2020.csv --era5 results/physics_evaluation_era5_2020.csv

  # Combined (auto-discovers CSVs in results/)
  python plot_timeseries.py combined
  python plot_timeseries.py combined --exclude aurora_2022 pangu_2022
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────

CONSERVATION_METRICS = {"dry_mass_Eg", "water_mass_kg", "total_energy_J", "mean_specific_humidity"}

METRICS = {
    "dry_mass_Eg":              "Dry Air Mass (Eg)",
    "water_mass_kg":            "Water Mass (kg)",
    "total_energy_J":           "Total Energy (J)",
    "mean_specific_humidity":   "Mean Specific Humidity (kg/kg)",
    "hydrostatic_rmse":         "Hydrostatic RMSE",
    "geostrophic_rmse":         "Geostrophic RMSE",
}

MODEL_STYLES = {
    "aurora_2022":    {"color": "#1f77b4", "marker": "o"},
    "pangu_2020":     {"color": "#ff7f0e", "marker": "s"},
    "pangu_2022":     {"color": "#ff7f0e", "marker": "s"},
    "fuxi_2020":      {"color": "#2ca02c", "marker": "^"},
    "graphcast_2020": {"color": "#d62728", "marker": "D"},
    "neuralgcm_2020": {"color": "#9467bd", "marker": "v"},
    "hres_2020":      {"color": "#8c564b", "marker": "P"},
    "ifs_ens_2020":   {"color": "#e377c2", "marker": "X"},
    "gencast_2020":   {"color": "#17becf", "marker": "x"},
}

NICE_NAMES = {
    "aurora_2022":    "Aurora",
    "pangu_2020":     "Pangu-Weather",
    "pangu_2022":     "Pangu-Weather (2022)",
    "fuxi_2020":      "FuXi",
    "graphcast_2020": "GraphCast",
    "neuralgcm_2020": "NeuralGCM",
    "hres_2020":      "HRES",
    "ifs_ens_2020":   "IFS-ENS",
    "gencast_2020":   "GenCast",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_model_name(path: str) -> str:
    """Extract model tag (with year) from filename like physics_timeseries_aurora_2022.csv."""
    name = Path(path).stem.replace("physics_timeseries_", "")
    # Strip _ifs suffix for consistent model names
    if name.endswith("_ifs"):
        name = name[:-4]
    return name


def _infer_model_label(path: str) -> str:
    """Human-readable label inferred from path."""
    tag = _extract_model_name(path)
    return NICE_NAMES.get(tag, tag)


def _infer_n_dates(path: str) -> int:
    """Guess the number of dates from the year in the filename."""
    m = re.search(r"(\d{4})", Path(path).stem)
    if m:
        year = int(m.group(1))
        import calendar
        return 366 if calendar.isleap(year) else 365
    return 365


def _compute_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_rel columns for conservation metrics (% change from first timestep)."""
    first_hour = df["forecast_hour"].min()
    for col in CONSERVATION_METRICS:
        if col not in df.columns or df[col].isna().all():
            continue
        t0_map = (
            df.loc[df["forecast_hour"] == first_hour]
            .drop_duplicates(subset=["date"])
            .set_index("date")[col]
        )
        mapped = df["date"].map(t0_map)
        df[f"{col}_rel"] = (df[col] - mapped) / mapped * 100
    return df


def _load_era5_rmse(results_dir: Path) -> dict[str, dict[int, float]]:
    """Load ERA5 intrinsic RMSE from physics_evaluation_era5_*.csv files."""
    era5_raw: dict[str, dict[int, list[float]]] = {}
    for p in sorted(results_dir.glob("physics_evaluation_era5_*.csv")):
        edf = pd.read_csv(p)
        for m in ("geostrophic_rmse", "hydrostatic_rmse"):
            sub = edf[edf["metric_name"] == m]
            if "era5_value" not in sub.columns:
                continue
            sub = sub.dropna(subset=["era5_value"])
            for lt, grp in sub.groupby("lead_time_hours"):
                era5_raw.setdefault(m, {}).setdefault(int(lt), []).extend(
                    grp["era5_value"].tolist()
                )
    return {
        m: {lt: float(np.mean(vals)) for lt, vals in lt_dict.items()}
        for m, lt_dict in era5_raw.items()
    }


def _load_era5_from_summary(summary_path: str) -> dict[str, dict[int, float]]:
    """Load ERA5 baselines from a summary/evaluation CSV with era5_value column."""
    summ_df = pd.read_csv(summary_path)
    if "era5_value" not in summ_df.columns:
        return {}
    out: dict[str, dict[int, float]] = {}
    for metric in ("hydrostatic_rmse", "geostrophic_rmse"):
        sub = summ_df[
            (summ_df["metric_name"] == metric) & summ_df["era5_value"].notna()
        ]
        if not sub.empty:
            out[metric] = (
                sub.groupby("lead_time_hours")["era5_value"].mean().to_dict()
            )
    return out


# ── Single-model plotting ────────────────────────────────────────────────────

def plot_single(csv_path: str, era5_csv: str | None = None):
    """Plot per-metric timeseries for one model, with optional ERA5 baseline."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "forecast_hour"])
    df = _compute_relative(df)

    era5_means = _load_era5_from_summary(era5_csv) if era5_csv else {}

    model_label = _infer_model_label(csv_path)
    n_dates = _infer_n_dates(csv_path)
    tag = _extract_model_name(csv_path)
    outdir = Path(csv_path).parent / f"plots_{tag}"
    outdir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    max_hour = int(df["forecast_hour"].max())

    for col, title in METRICS.items():
        if col not in df.columns:
            continue

        # Absolute plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=df, x="forecast_hour", y=col, errorbar=("ci", 95),
            marker="o", markersize=6, color="royalblue",
            label=f"{model_label} Mean ± 95% CI ({n_dates} dates)", zorder=3, ax=ax,
        )
        if col in era5_means:
            lts = sorted(era5_means[col])
            ax.plot(lts, [era5_means[col][lt] for lt in lts],
                    color="black", linewidth=2, linestyle="--",
                    marker="*", markersize=10, label="ERA5 (intrinsic)",
                    zorder=5, alpha=0.8)

        ax.set_title(f"{title} over Forecast Horizon (1 yr average)", fontsize=14, pad=15)
        ax.set_xlabel("Forecast Hour", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(range(0, max_hour + 1, 24))
        ax.legend()
        fig.tight_layout()
        outfile = outdir / f"{col}_timeseries.png"
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f"Saved {outfile}")

        # Relative plot for conservation metrics
        if col in CONSERVATION_METRICS:
            rel_col = f"{col}_rel"
            if rel_col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=df, x="forecast_hour", y=rel_col, errorbar=("ci", 95),
                marker="o", markersize=6, color="royalblue",
                label=f"{model_label} Mean ± 95% CI ({n_dates} dates)", zorder=3, ax=ax,
            )
            ax.axhline(y=0, color="grey", linestyle="-", linewidth=1, alpha=0.5)
            ax.set_title(f"{title} — Relative Change from t₀ (%, shading = 95% CI)", fontsize=14, pad=15)
            ax.set_xlabel("Forecast Hour", fontsize=12)
            ax.set_ylabel("Relative Change (%)", fontsize=12)
            ax.set_xticks(range(0, max_hour + 1, 24))
            ax.legend()
            fig.tight_layout()
            outfile = outdir / f"{col}_relative_timeseries.png"
            fig.savefig(outfile, dpi=300)
            plt.close(fig)
            print(f"Saved {outfile}")


# ── Combined multi-model plotting ────────────────────────────────────────────

def _load_and_preagg(csv_paths: list[str]) -> dict[str, pd.DataFrame]:
    """Load CSVs and pre-aggregate to mean/std per forecast_hour (fast)."""
    models: dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        model = _extract_model_name(path)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date", "forecast_hour"])
        df = _compute_relative(df)

        # Only keep numeric columns for aggregation
        value_cols = [c for c in df.columns if c not in ("date", "forecast_hour") and df[c].dtype in ["float64", "float32", "int64", "int32"]]
        agg = df.groupby("forecast_hour")[value_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["forecast_hour"] + [f"{c}_{s}" for c, s in agg.columns[1:]]
        models[model] = agg
    return models


def plot_combined(csv_paths: list[str], results_dir: Path, ifs_mode: bool = False):
    """Create combined multi-model overlay plots."""
    models = _load_and_preagg(csv_paths)
    era5_rmse = _load_era5_rmse(results_dir) if not ifs_mode else None
    
    # Use different output folder for IFS mode
    outdir = results_dir / ("plots_combined_IFS" if ifs_mode else "plots_combined")
    outdir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    max_hour = max(df["forecast_hour"].max() for df in models.values())

    for col, title in METRICS.items():
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        has_data = {m: df for m, df in models.items()
                    if mean_col in df.columns and not df[mean_col].isna().all()}
        if not has_data:
            continue

        # Absolute plot
        fig, ax = plt.subplots(figsize=(13, 6))
        for model, df in sorted(has_data.items()):
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            nice = NICE_NAMES.get(model, model)
            x = df["forecast_hour"].values
            y = df[mean_col].values
            yerr = df[std_col].values if std_col in df.columns else None
            ax.plot(x, y, marker=style["marker"], markersize=6,
                    color=style["color"], label=nice, zorder=3)
            if yerr is not None:
                ax.fill_between(x, y - yerr, y + yerr,
                                color=style["color"], alpha=0.15, zorder=2)

        if era5_rmse and col in era5_rmse:
            era5_lts = sorted(era5_rmse[col])
            ax.plot(era5_lts, [era5_rmse[col][lt] for lt in era5_lts],
                    color="black", linewidth=2, linestyle="--",
                    marker="*", markersize=10, label="ERA5 (intrinsic)",
                    zorder=5, alpha=0.8)

        ax.set_title(f"{title} over Forecast Horizon (shading = ±1σ)", fontsize=14, pad=15)
        ax.set_xlabel("Forecast Hour", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(range(0, int(max_hour) + 1, 24))
        ax.legend(fontsize=10)
        fig.tight_layout()
        outfile = outdir / f"{col}_combined.png"
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f"Saved {outfile}")

        # Relative plot for conservation metrics
        if col in CONSERVATION_METRICS:
            rel_mean = f"{col}_rel_mean"
            rel_std = f"{col}_rel_std"
            has_rel = {m: df for m, df in models.items()
                       if rel_mean in df.columns and not df[rel_mean].isna().all()}
            if not has_rel:
                continue

            fig, ax = plt.subplots(figsize=(13, 6))
            for model, df in sorted(has_rel.items()):
                style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
                nice = NICE_NAMES.get(model, model)
                x = df["forecast_hour"].values
                y = df[rel_mean].values
                yerr = df[rel_std].values if rel_std in df.columns else None
                ax.plot(x, y, marker=style["marker"], markersize=6,
                        color=style["color"], label=nice, zorder=3)
                if yerr is not None:
                    ax.fill_between(x, y - yerr, y + yerr,
                                    color=style["color"], alpha=0.15, zorder=2)

            ax.axhline(y=0, color="grey", linestyle="-", linewidth=1, alpha=0.5)
            ax.set_title(f"{title} — Relative Change from t₀ (%, shading = ±1σ)", fontsize=14, pad=15)
            ax.set_xlabel("Forecast Hour", fontsize=12)
            ax.set_ylabel("Relative Change (%)", fontsize=12)
            ax.set_xticks(range(0, int(max_hour) + 1, 24))
            ax.legend(fontsize=10)
            fig.tight_layout()
            outfile = outdir / f"{col}_relative_combined.png"
            fig.savefig(outfile, dpi=300)
            plt.close(fig)
            print(f"Saved {outfile}")

    print(f"\nAll combined plots saved to {outdir}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Timeseries plots for physics evaluation metrics."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- single ---
    sp = sub.add_parser("single", help="Plot one model's timeseries")
    sp.add_argument("csv", help="Path to physics_timeseries_<model>_<year>.csv")
    sp.add_argument("--era5", default=None,
                    help="ERA5 evaluation/summary CSV for baseline overlay")

    # --- combined ---
    cp = sub.add_parser("combined", help="Multi-model overlay plots")
    cp.add_argument("--results-dir", default=None,
                    help="Directory containing physics_timeseries_*.csv "
                         "(default: thesis/results/)")
    cp.add_argument("--exclude", nargs="*", default=[],
                    help="Model tags to exclude (e.g. aurora_2022 pangu_2022)")
    cp.add_argument("--ifs", action="store_true",
                    help="Use IFS HRES comparison files (*_ifs.csv) instead of ERA5")

    args = parser.parse_args()

    if args.mode == "single":
        plot_single(args.csv, era5_csv=args.era5)

    elif args.mode == "combined":
        results_dir = Path(args.results_dir) if args.results_dir else (
            Path(__file__).resolve().parent.parent / "results"
        )
        exclude = set(args.exclude)
        csv_paths = sorted(glob.glob(str(results_dir / "physics_timeseries_*.csv")))
        
        # Filter based on IFS flag
        if args.ifs:
            csv_paths = [p for p in csv_paths if p.endswith("_ifs.csv")]
        else:
            csv_paths = [p for p in csv_paths if not p.endswith("_ifs.csv")]
        
        csv_paths = [p for p in csv_paths if _extract_model_name(p) not in exclude]
        if not csv_paths:
            print(f"No physics_timeseries_*.csv found in {results_dir}")
            return
        print(f"Found {len(csv_paths)} timeseries files:")
        for p in csv_paths:
            print(f"  {Path(p).name}")
        print()
        plot_combined(csv_paths, results_dir, ifs_mode=args.ifs)


if __name__ == "__main__":
    main()
