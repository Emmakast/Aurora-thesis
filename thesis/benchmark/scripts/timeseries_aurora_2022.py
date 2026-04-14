#!/usr/bin/env python3
"""
Plot timeseries of Aurora 2022 forecast errors vs lead time.

Reads evaluation results from evaluate_aurora_2022.py and creates:
1. Error evolution plots (RMSE/ACC vs lead time)
2. Comparison with WB2 benchmark models
3. Per-variable skill degradation curves

Usage:
    python timeseries_aurora_2022.py
    python timeseries_aurora_2022.py --eval-csv path/to/evaluation.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

warnings.filterwarnings("ignore")

RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"
PLOTS_DIR = RESULTS_DIR / "plots_aurora_2022"

# WB2 benchmark paths (for comparison)
WB2_BENCHMARKS = {
    "pangu": "gs://weatherbench2/benchmark_results/pangu_hres_init_vs_era5_1440x721_2020.nc",
    "graphcast": "gs://weatherbench2/benchmark_results/graphcast_hres_init_vs_era5_1440x721_2020.nc",
    "fuxi": "gs://weatherbench2/benchmark_results/fuxi_vs_era5_1440x721_2020.nc",
    "hres": "gs://weatherbench2/benchmark_results/hres_vs_era5_1440x721_2020.nc",
}


def load_aurora_evaluation(csv_path):
    """Load Aurora evaluation results."""
    df = pd.read_csv(csv_path)
    return df


def load_wb2_benchmark(model_name, metric_type="rmse"):
    """Load WB2 benchmark results for a model."""
    import gcsfs
    import tempfile
    
    path = WB2_BENCHMARKS.get(model_name)
    if not path:
        return None
    
    fs = gcsfs.GCSFileSystem(token="anon")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            tmp = f.name
        fs.get(path, tmp)
        
        ds = xr.open_dataset(tmp, decode_timedelta=False)
        
        # Extract lead times
        lead_times = ds.lead_time.values
        
        # Handle lead time units
        if isinstance(lead_times[0], np.timedelta64):
            lead_hours = [float(lt / np.timedelta64(1, "h")) for lt in lead_times]
        else:
            # Assume nanoseconds if values are very large
            if np.max(lead_times) > 1e10:
                lead_hours = [float(lt / 3600e9) for lt in lead_times]
            else:
                lead_hours = list(lead_times)
        
        # Extract metrics
        results = {"lead_hours": lead_hours}
        
        if f"{metric_type}.geopotential" in ds:
            z500 = ds[f"{metric_type}.geopotential"].sel(level=500).values
            results["z500"] = z500
        
        if f"{metric_type}.temperature" in ds:
            t850 = ds[f"{metric_type}.temperature"].sel(level=850).values
            results["t850"] = t850
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"  ⚠ Could not load {model_name}: {e}")
        return None


def plot_rmse_vs_leadtime(df_aurora, output_dir):
    """Plot RMSE evolution vs lead time."""
    sns.set_theme(style="whitegrid")
    
    # Group by lead time and compute mean RMSE
    metrics_to_plot = [
        ("z_500hPa_rmse", "Geopotential 500 hPa (m²/s²)"),
        ("t_850hPa_rmse", "Temperature 850 hPa (K)"),
        ("2t_rmse", "2m Temperature (K)"),
        ("msl_rmse", "Mean Sea Level Pressure (Pa)"),
    ]
    
    for metric, label in metrics_to_plot:
        if metric not in df_aurora.columns:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aurora 2022 - mean and std by lead time
        grouped = df_aurora.groupby("lead_hours")[metric]
        means = grouped.mean()
        stds = grouped.std()
        
        ax.plot(means.index, means.values, "o-", color="tab:blue", linewidth=2, 
                markersize=6, label="Aurora (2022)")
        ax.fill_between(means.index, means - stds, means + stds, 
                       color="tab:blue", alpha=0.2)
        
        ax.set_xlabel("Lead Time (hours)", fontsize=12)
        ax.set_ylabel(f"RMSE - {label}", fontsize=12)
        ax.set_title(f"Aurora 2022 - {label} RMSE vs Lead Time", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / f"aurora_2022_rmse_{metric.replace('_rmse', '')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path.name}")


def plot_acc_vs_leadtime(df_aurora, output_dir):
    """Plot ACC evolution vs lead time."""
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = [
        ("z_500hPa_acc", "Geopotential 500 hPa"),
        ("t_850hPa_acc", "Temperature 850 hPa"),
    ]
    
    for metric, label in metrics_to_plot:
        if metric not in df_aurora.columns:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped = df_aurora.groupby("lead_hours")[metric]
        means = grouped.mean()
        stds = grouped.std()
        
        ax.plot(means.index, means.values, "o-", color="tab:green", linewidth=2,
                markersize=6, label="Aurora (2022)")
        ax.fill_between(means.index, means - stds, means + stds,
                       color="tab:green", alpha=0.2)
        
        # Add reference line at ACC = 0.6 (commonly used threshold)
        ax.axhline(0.6, color="red", linestyle="--", alpha=0.7, label="ACC = 0.6 threshold")
        
        ax.set_xlabel("Lead Time (hours)", fontsize=12)
        ax.set_ylabel(f"ACC - {label}", fontsize=12)
        ax.set_title(f"Aurora 2022 - {label} ACC vs Lead Time", fontsize=14)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        out_path = output_dir / f"aurora_2022_acc_{metric.replace('_acc', '')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path.name}")


def plot_comparison_with_wb2(df_aurora, output_dir):
    """Plot Aurora 2022 vs WB2 benchmark models."""
    sns.set_theme(style="whitegrid")
    
    # Load WB2 benchmarks
    print("\n  Loading WB2 benchmarks for comparison...")
    wb2_data = {}
    for model in WB2_BENCHMARKS:
        df_model = load_wb2_benchmark(model)
        if df_model is not None:
            wb2_data[model] = df_model
            print(f"    ✓ {model}")
    
    # Prepare Aurora data
    aurora_grouped = df_aurora.groupby("lead_hours").mean(numeric_only=True).reset_index()
    
    # Colors for models
    colors = {
        "aurora": "tab:blue",
        "pangu": "tab:orange",
        "graphcast": "tab:green",
        "fuxi": "tab:red",
        "hres": "tab:purple",
    }
    
    # Plot Z500 RMSE comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Aurora
    if "z_500hPa_rmse" in aurora_grouped.columns:
        ax.plot(aurora_grouped["lead_hours"], aurora_grouped["z_500hPa_rmse"],
                "o-", color=colors["aurora"], linewidth=2, markersize=5, label="Aurora (2022)")
    
    # WB2 models
    for model, df_model in wb2_data.items():
        if "z500" in df_model.columns:
            ax.plot(df_model["lead_hours"], df_model["z500"],
                    "s--", color=colors.get(model, "gray"), linewidth=1.5, 
                    markersize=4, alpha=0.8, label=f"{model.title()} (2020)")
    
    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("Z500 RMSE (m²/s²)", fontsize=12)
    ax.set_title("Geopotential 500 hPa RMSE: Aurora 2022 vs WB2 Benchmarks", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)
    
    plt.tight_layout()
    out_path = output_dir / "aurora_2022_vs_wb2_z500_rmse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")
    
    # Plot T850 RMSE comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if "t_850hPa_rmse" in aurora_grouped.columns:
        ax.plot(aurora_grouped["lead_hours"], aurora_grouped["t_850hPa_rmse"],
                "o-", color=colors["aurora"], linewidth=2, markersize=5, label="Aurora (2022)")
    
    for model, df_model in wb2_data.items():
        if "t850" in df_model.columns:
            ax.plot(df_model["lead_hours"], df_model["t850"],
                    "s--", color=colors.get(model, "gray"), linewidth=1.5,
                    markersize=4, alpha=0.8, label=f"{model.title()} (2020)")
    
    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("T850 RMSE (K)", fontsize=12)
    ax.set_title("Temperature 850 hPa RMSE: Aurora 2022 vs WB2 Benchmarks", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)
    
    plt.tight_layout()
    out_path = output_dir / "aurora_2022_vs_wb2_t850_rmse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_daily_variability(df_aurora, output_dir):
    """Plot daily variability in forecast errors."""
    sns.set_theme(style="whitegrid")
    
    # Pick a few representative lead times
    lead_times = [24, 72, 120, 168, 240]
    available_leads = df_aurora["lead_hours"].unique()
    lead_times = [lt for lt in lead_times if lt in available_leads]
    
    if not lead_times:
        lead_times = sorted(available_leads)[:5]
    
    fig, axes = plt.subplots(1, len(lead_times), figsize=(4 * len(lead_times), 5))
    if len(lead_times) == 1:
        axes = [axes]
    
    metric = "z_500hPa_rmse"
    if metric not in df_aurora.columns:
        return
    
    for ax, lead in zip(axes, lead_times):
        sub = df_aurora[df_aurora["lead_hours"] == lead][metric].dropna()
        ax.hist(sub, bins=20, color="tab:blue", alpha=0.7, edgecolor="black")
        ax.axvline(sub.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {sub.mean():.1f}")
        ax.set_xlabel("Z500 RMSE", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"+{lead}h", fontsize=12)
        ax.legend(fontsize=8)
    
    plt.suptitle("Daily Variability in Z500 RMSE", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "aurora_2022_daily_variability.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def create_summary_table(df_aurora, output_dir):
    """Create summary statistics table."""
    # Group by lead time
    lead_groups = df_aurora.groupby("lead_hours")
    
    metrics = ["z_500hPa_rmse", "t_850hPa_rmse", "2t_rmse", "z_500hPa_acc", "t_850hPa_acc"]
    metrics = [m for m in metrics if m in df_aurora.columns]
    
    summary = lead_groups[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    
    # Flatten column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    
    summary.to_csv(output_dir / "aurora_2022_summary_stats.csv", index=False)
    print(f"  Saved aurora_2022_summary_stats.csv")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot Aurora 2022 timeseries")
    parser.add_argument(
        "--eval-csv", type=str, default=None,
        help="Path to evaluation CSV from evaluate_aurora_2022.py"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots"
    )
    args = parser.parse_args()
    
    # Paths
    eval_csv = Path(args.eval_csv) if args.eval_csv else RESULTS_DIR / "aurora_2022_evaluation.csv"
    output_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  AURORA 2022 TIMESERIES PLOTS")
    print("=" * 70)
    
    # Load evaluation data
    print(f"\n[1/5] Loading evaluation data from {eval_csv}...")
    if not eval_csv.exists():
        print(f"  ⚠ Evaluation file not found: {eval_csv}")
        print("  Run evaluate_aurora_2022.py first.")
        return
    
    df = load_aurora_evaluation(eval_csv)
    print(f"  Loaded {len(df)} evaluations")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Lead times: {sorted(df['lead_hours'].unique())}")
    
    # Generate plots
    print("\n[2/5] Plotting RMSE vs lead time...")
    plot_rmse_vs_leadtime(df, output_dir)
    
    print("\n[3/5] Plotting ACC vs lead time...")
    plot_acc_vs_leadtime(df, output_dir)
    
    print("\n[4/5] Plotting comparison with WB2 benchmarks...")
    plot_comparison_with_wb2(df, output_dir)
    
    print("\n[5/5] Creating summary statistics...")
    plot_daily_variability(df, output_dir)
    summary = create_summary_table(df, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    for lead in sorted(df["lead_hours"].unique())[:5]:  # Show first 5
        sub = df[df["lead_hours"] == lead]
        line = f"  Lead +{lead:3d}h:"
        if "z_500hPa_rmse" in sub.columns:
            line += f"  Z500 RMSE={sub['z_500hPa_rmse'].mean():.1f}"
        if "t_850hPa_rmse" in sub.columns:
            line += f"  T850 RMSE={sub['t_850hPa_rmse'].mean():.2f}"
        if "z_500hPa_acc" in sub.columns:
            line += f"  Z500 ACC={sub['z_500hPa_acc'].mean():.3f}"
        print(line)
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
