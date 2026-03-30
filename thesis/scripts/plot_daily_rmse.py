#!/usr/bin/env python
"""
Plot daily RMSE from pre-computed CSV files.

Creates:
  1. Combined Z500/T850 RMSE boxplots per lead time
  2. Time series RMSE evolution per model

Usage:
    python plot_daily_rmse.py
    python plot_daily_rmse.py --models pangu graphcast hres
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAIN_RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

MODEL_STYLES = {
    "pangu":     {"color": "#ff7f0e", "label": "Pangu-Weather"},
    "fuxi":      {"color": "#2ca02c", "label": "FuXi"},
    "graphcast": {"color": "#d62728", "label": "GraphCast"},
    "neuralgcm": {"color": "#9467bd", "label": "NeuralGCM"},
    "hres":      {"color": "#8c564b", "label": "HRES"},
}

MODEL_ORDER = ["hres", "pangu", "graphcast", "fuxi", "neuralgcm"]


def load_rmse_data(results_dir: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Load all daily_rmse CSVs and combine."""
    dfs = []
    for csv_path in sorted(results_dir.glob("daily_rmse_*.csv")):
        model = csv_path.stem.replace("daily_rmse_", "").rsplit("_", 1)[0]
        if models and model not in models:
            continue
        df = pd.read_csv(csv_path)
        df["model"] = model
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No daily_rmse_*.csv files found in {results_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


def plot_rmse_boxplot(df: pd.DataFrame, outdir: Path):
    """Boxplot of RMSE per model, per lead time."""
    sns.set_theme(style="whitegrid")
    
    lead_times = sorted(df["lead_time_hours"].unique())
    
    for metric, ylabel, units in [
        ("z500_rmse", "Z500 RMSE", "m²/s²"),
        ("t850_rmse", "T850 RMSE", "K"),
    ]:
        fig, axes = plt.subplots(1, len(lead_times), figsize=(5 * len(lead_times), 6), sharey=True)
        if len(lead_times) == 1:
            axes = [axes]
        
        for ax, lt in zip(axes, lead_times):
            sub = df[df["lead_time_hours"] == lt].copy()
            
            # Order by median for cleaner comparison
            model_order = [m for m in MODEL_ORDER if m in sub["model"].unique()]
            
            palette = {m: MODEL_STYLES.get(m, {}).get("color", "grey") for m in model_order}
            
            sns.boxplot(
                data=sub, x="model", y=metric, order=model_order,
                hue="model", hue_order=model_order, palette=palette, 
                ax=ax, showfliers=False, legend=False
            )
            ax.set_title(f"Lead +{lt}h", fontsize=12)
            ax.set_xlabel("")
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels([MODEL_STYLES.get(m, {}).get("label", m) for m in model_order], rotation=30, ha="right")
            
        axes[0].set_ylabel(f"{ylabel} ({units})", fontsize=11)
        fig.suptitle(f"{ylabel} Distribution by Lead Time (2020)", fontsize=14)
        fig.tight_layout()
        
        outfile = outdir / f"{metric}_boxplot.png"
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f"Saved {outfile}")


def plot_rmse_timeseries(df: pd.DataFrame, outdir: Path):
    """Time-series RMSE per model over the year."""
    sns.set_theme(style="whitegrid")
    
    lead_times = sorted(df["lead_time_hours"].unique())
    
    for metric, ylabel, units in [
        ("z500_rmse", "Z500 RMSE", "m²/s²"),
        ("t850_rmse", "T850 RMSE", "K"),
    ]:
        for lt in lead_times:
            fig, ax = plt.subplots(figsize=(14, 5))
            sub = df[df["lead_time_hours"] == lt].copy()
            
            # Average per date if multiple init times
            agg = sub.groupby(["date", "model"])[metric].mean().reset_index()
            
            for model in MODEL_ORDER:
                if model not in agg["model"].unique():
                    continue
                m_data = agg[agg["model"] == model].sort_values("date")
                style = MODEL_STYLES.get(model, {"color": "grey", "label": model})
                ax.plot(m_data["date"], m_data[metric], 
                        color=style["color"], label=style["label"], alpha=0.8, linewidth=1)
            
            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel(f"{ylabel} ({units})", fontsize=11)
            ax.set_title(f"{ylabel} Over 2020 — Lead +{lt}h", fontsize=13)
            ax.legend(loc="upper right", fontsize=10)
            fig.tight_layout()
            
            outfile = outdir / f"{metric}_timeseries_lead{lt}h.png"
            fig.savefig(outfile, dpi=300)
            plt.close(fig)
            print(f"Saved {outfile}")


def plot_rmse_summary_bars(df: pd.DataFrame, outdir: Path):
    """Bar chart of mean RMSE per model at each lead time."""
    sns.set_theme(style="whitegrid")
    
    lead_times = sorted(df["lead_time_hours"].unique())
    
    for metric, ylabel, units in [
        ("z500_rmse", "Z500 RMSE", "m²/s²"),
        ("t850_rmse", "T850 RMSE", "K"),
    ]:
        # Aggregate: mean and std per model per lead time
        agg = df.groupby(["model", "lead_time_hours"])[metric].agg(["mean", "std"]).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar_width = 0.15
        x_base = range(len(lead_times))
        
        models_present = [m for m in MODEL_ORDER if m in agg["model"].unique()]
        
        for i, model in enumerate(models_present):
            m_data = agg[agg["model"] == model].set_index("lead_time_hours")
            means = [m_data.loc[lt, "mean"] if lt in m_data.index else 0 for lt in lead_times]
            stds = [m_data.loc[lt, "std"] if lt in m_data.index else 0 for lt in lead_times]
            x = [xi + i * bar_width for xi in x_base]
            
            style = MODEL_STYLES.get(model, {"color": "grey", "label": model})
            ax.bar(x, means, bar_width, label=style["label"], color=style["color"], 
                   yerr=stds, capsize=3, alpha=0.85)
        
        ax.set_xlabel("Lead Time (hours)", fontsize=11)
        ax.set_ylabel(f"Mean {ylabel} ({units})", fontsize=11)
        ax.set_title(f"Mean {ylabel} by Lead Time (2020, error bars = ±1σ)", fontsize=13)
        ax.set_xticks([xi + bar_width * (len(models_present) - 1) / 2 for xi in x_base])
        ax.set_xticklabels([f"+{lt}h" for lt in lead_times])
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        
        outfile = outdir / f"{metric}_summary_bars.png"
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Plot daily RMSE from CSV files")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to include (default: all)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing daily_rmse_*.csv files")
    args = parser.parse_args()
    
    # Try worktree results dir first, then main results dir
    results_dir = Path(args.results_dir) if args.results_dir else MAIN_RESULTS_DIR
    if not any(results_dir.glob("daily_rmse_*.csv")):
        results_dir = RESULTS_DIR
    
    print(f"Loading RMSE data from {results_dir}...")
    df = load_rmse_data(results_dir, args.models)
    print(f"  Loaded {len(df)} rows for models: {sorted(df['model'].unique())}")
    print(f"  Lead times: {sorted(df['lead_time_hours'].unique())}")
    
    outdir = results_dir / "plots_rmse"
    outdir.mkdir(exist_ok=True)
    
    plot_rmse_boxplot(df, outdir)
    plot_rmse_timeseries(df, outdir)
    plot_rmse_summary_bars(df, outdir)
    
    print(f"\nAll RMSE plots saved to {outdir}/")


if __name__ == "__main__":
    main()
