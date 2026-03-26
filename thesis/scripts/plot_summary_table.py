#!/usr/bin/env python3
"""
Render a colour-coded summary table comparing all models against HRES.

Blue = better than HRES, Red = worse than HRES, white = HRES (reference).
One table per lead time, saved as PNG.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────

MODELS = ["hres", "fuxi", "graphcast", "neuralgcm", "pangu"]
NICE = {
    "hres": "HRES", "fuxi": "FuXi", "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM", "pangu": "Pangu-Weather",
}

# Metrics to show (in order) and whether lower is better
METRICS = {
    "hydrostatic_rmse":             ("Hydrostatic\nRMSE (Δ ERA5)",     True),
    "geostrophic_rmse":             ("Geostrophic\nRMSE (Δ ERA5)",     True),
    "dry_mass_drift_pct_per_day":   ("Dry Mass Drift\n(%/day)",        True),  # closer to 0
    "water_mass_drift_pct_per_day": ("Water Mass Drift\n(%/day)",      True),
    "total_energy_drift_pct_per_day":("Energy Drift\n(%/day)",         True),
    "effective_resolution_km":      ("Effective Res.\n(km)",            True),
    "spectral_residual":            ("Spectral\nResidual",             True),
    "spectral_divergence":          ("Spectral\nDivergence",           True),
    "small_scale_ratio":            ("Small-Scale\nRatio",             False),  # closer to 1 is better, >1 good
}

# For drift metrics, "better" means closer to zero (use absolute value)
ABS_METRICS = {
    "dry_mass_drift_pct_per_day",
    "water_mass_drift_pct_per_day",
    "total_energy_drift_pct_per_day",
}


def load_summaries(results_dir: Path, ifs_mode: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    suffix = "_ifs" if ifs_mode else ""
    for m in MODELS:
        p = results_dir / f"physics_summary_{m}_2020{suffix}.csv"
        if p.exists():
            out[m] = pd.read_csv(p)
    return out


def get_value(df: pd.DataFrame, metric: str) -> float:
    """Extract the single value for a metric from a summary slice."""
    row = df[df["metric_name"] == metric]
    if row.empty:
        return np.nan
    r = row.iloc[0]
    # Prefer mean_value (= model−ERA5 diff for RMSE, raw model for others)
    val = r["mean_value"] if pd.notna(r.get("mean_value")) else r.get("mean_model")
    return float(val) if pd.notna(val) else np.nan


def fmt(val: float, metric: str) -> str:
    if np.isnan(val):
        return "—"
    if metric in ABS_METRICS:
        return f"{val:+.4f}"
    if "rmse" in metric:
        return f"{val:.2f}"
    if metric == "effective_resolution_km":
        return f"{val:.1f}"
    if metric == "small_scale_ratio":
        return f"{val:.3f}"
    return f"{val:.4f}"


def build_table_data(lead: int, summaries: dict[str, pd.DataFrame],
                     metrics_list: list[str]):
    """Build cell texts, colours, and values for one lead-time block.
    
    Colors: white = 0 (ideal), blue = positive, red = negative.
    For models that don't have this lead time, use the nearest available
    lead time's data marked with '***'.
    """
    blue = np.array([0.7, 0.85, 1.0])   # positive values
    red = np.array([1.0, 0.75, 0.75])   # negative values
    white = np.array([1.0, 1.0, 1.0])   # zero (ideal)
    hres_grey = np.array([0.94, 0.94, 0.94])

    n_metrics = len(metrics_list)

    # Get values — for models missing this lead, try nearest
    vals = np.full((len(MODELS), n_metrics), np.nan)
    is_approx = [False] * len(MODELS)  # track which models used a different lead
    for i, m in enumerate(MODELS):
        if m not in summaries:
            continue
        df = summaries[m]
        df_lt = df[df["lead_time_hours"] == lead]
        if df_lt.empty:
            # Find nearest lead time (for NeuralGCM 12h → 6h slot)
            avail = sorted(df["lead_time_hours"].unique())
            nearest = min(avail, key=lambda x: abs(x - lead))
            df_lt = df[df["lead_time_hours"] == nearest]
            is_approx[i] = True
        for j, metric in enumerate(metrics_list):
            vals[i, j] = get_value(df_lt, metric)

    # For each metric, find the max absolute value across all models for scaling
    max_abs_per_metric = np.nanmax(np.abs(vals), axis=0)
    max_abs_per_metric[max_abs_per_metric == 0] = 1.0  # avoid division by zero

    cell_texts = []
    cell_colors = []
    for i, m in enumerate(MODELS):
        row_texts = []
        row_colors = []
        for j, metric in enumerate(metrics_list):
            v = vals[i, j]

            if is_approx[i] and np.isnan(v):
                text = "***"
            elif is_approx[i]:
                text = fmt(v, metric) + " ***"
            else:
                text = fmt(v, metric)
            row_texts.append(text)

            if np.isnan(v):
                row_colors.append(white)
                continue

            # Color based on sign: white=0, blue=positive, red=negative
            # Intensity scales with |value| / max_abs for this metric
            intensity = min(abs(v) / max_abs_per_metric[j], 1.0) * 0.8  # cap at 0.8 for readability
            
            if v > 0:
                color = white * (1 - intensity) + blue * intensity
            elif v < 0:
                color = white * (1 - intensity) + red * intensity
            else:
                color = white
            row_colors.append(color)

        cell_texts.append(row_texts)
        cell_colors.append(row_colors)

    return cell_texts, cell_colors


def render_combined_table(leads: list[int], summaries: dict[str, pd.DataFrame],
                          outdir: Path):
    """Render one figure with all lead times stacked vertically."""
    # Use union of all metrics across all leads
    metrics_list = list(METRICS.keys())

    n_metrics = len(metrics_list)
    n_models = len(MODELS)
    col_labels = [METRICS[m][0] for m in metrics_list]
    model_labels = [NICE.get(m, m) for m in MODELS]

    n_leads = len(leads)
    # Total rows: header + (lead-time label + n_models) per lead
    total_rows = n_leads * (n_models + 1)  # +1 for sub-header per lead

    fig_w = max(2.0 * n_metrics, 16)
    fig_h = 0.48 * total_rows + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Build the full cell matrix
    all_texts = []
    all_colors = []
    all_row_labels = []

    hres_grey = np.array([0.94, 0.94, 0.94])
    header_color = np.array([0.82, 0.82, 0.82])

    for lead in leads:
        lead_str = f"{lead}h" if lead < 24 else f"{lead // 24}d ({lead}h)"

        # Sub-header row for this lead time
        all_texts.append([""] * n_metrics)
        all_colors.append([tuple(header_color)] * n_metrics)
        all_row_labels.append(f"  Lead Time: {lead_str}")

        texts, colors = build_table_data(lead, summaries, metrics_list)
        for i in range(n_models):
            all_texts.append(texts[i])
            all_colors.append([tuple(c) for c in colors[i]])
            all_row_labels.append(model_labels[i])

    table = ax.table(
        cellText=all_texts,
        rowLabels=all_row_labels,
        colLabels=col_labels,
        cellColours=all_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    # Style column headers
    for j in range(n_metrics):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_facecolor("#d0d0d0")

    # Style rows
    row_idx = 0
    for lead in leads:
        row_idx += 1  # sub-header
        for j in range(n_metrics):
            table[row_idx, j].set_facecolor(tuple(header_color))
        # Row label for sub-header — bold, left-aligned
        label_cell = table[row_idx, -1]
        label_cell.set_text_props(fontweight="bold", fontsize=10)
        label_cell.set_facecolor(tuple(header_color))

        for i in range(n_models):
            row_idx += 1
            label_cell = table[row_idx, -1]
            label_cell.set_text_props(fontweight="bold", fontsize=9)
            if MODELS[i] == "hres":
                label_cell.set_facecolor(tuple(hres_grey))

    ax.set_title(
        "Physics Metrics Summary — All Lead Times\n"
        "(White = 0, Blue = positive, Red = negative, *** = nearest available lead time)",
        fontsize=13, fontweight="bold", pad=12,
    )
    out = outdir / "summary_table_combined.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")

    # Also save individual tables
    for lead in leads:
        lead_str = f"{lead}h"
        texts, colors = build_table_data(lead, summaries, metrics_list)
        fig2, ax2 = plt.subplots(figsize=(fig_w, max(0.55 * n_models + 1.5, 4)))
        ax2.axis("off")
        t2 = ax2.table(
            cellText=texts,
            rowLabels=model_labels,
            colLabels=col_labels,
            cellColours=[[tuple(c) for c in row] for row in colors],
            loc="center", cellLoc="center",
        )
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1.0, 1.8)
        for j in range(n_metrics):
            t2[0, j].set_text_props(fontweight="bold", fontsize=9)
            t2[0, j].set_facecolor("#e0e0e0")
        for i in range(n_models):
            t2[i + 1, -1].set_text_props(fontweight="bold")
            if MODELS[i] == "hres":
                t2[i + 1, -1].set_facecolor(tuple(hres_grey))
        lead_nice = f"{lead}h" if lead < 24 else f"{lead // 24}d ({lead}h)"
        ax2.set_title(
            f"Physics Metrics Summary — Lead Time {lead_nice}\n"
            f"(White = 0, Blue = positive, Red = negative, *** = nearest available lead time)",
            fontsize=13, fontweight="bold", pad=20,
        )
        fig2.tight_layout()
        out2 = outdir / f"summary_table_{lead_str}.png"
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {out2}")


def render_combined_table_by_model(leads: list[int], summaries: dict[str, pd.DataFrame],
                                    outdir: Path):
    """Render one figure with all models stacked vertically (grouped by model)."""
    metrics_list = list(METRICS.keys())

    n_metrics = len(metrics_list)
    n_leads = len(leads)
    col_labels = [METRICS[m][0] for m in metrics_list]

    # Total rows: (model label + n_leads) per model
    total_rows = len(MODELS) * (n_leads + 1)

    fig_w = max(2.0 * n_metrics, 16)
    fig_h = 0.48 * total_rows + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    all_texts = []
    all_colors = []
    all_row_labels = []

    hres_grey = np.array([0.94, 0.94, 0.94])
    header_color = np.array([0.82, 0.82, 0.82])

    for model in MODELS:
        model_label = NICE.get(model, model)

        # Sub-header row for this model
        all_texts.append([""] * n_metrics)
        all_colors.append([tuple(header_color)] * n_metrics)
        all_row_labels.append(f"  {model_label}")

        for lead in leads:
            lead_str = f"{lead}h" if lead < 24 else f"{lead // 24}d ({lead}h)"
            
            # Get data for this single model at this lead time
            texts, colors = build_table_data(lead, summaries, metrics_list)
            model_idx = MODELS.index(model)
            
            all_texts.append(texts[model_idx])
            all_colors.append([tuple(c) for c in colors[model_idx]])
            all_row_labels.append(f"    {lead_str}")

    table = ax.table(
        cellText=all_texts,
        rowLabels=all_row_labels,
        colLabels=col_labels,
        cellColours=all_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    # Style column headers
    for j in range(n_metrics):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_facecolor("#d0d0d0")

    # Style rows
    row_idx = 0
    for model in MODELS:
        row_idx += 1  # sub-header (model name)
        for j in range(n_metrics):
            table[row_idx, j].set_facecolor(tuple(header_color))
        label_cell = table[row_idx, -1]
        label_cell.set_text_props(fontweight="bold", fontsize=10)
        label_cell.set_facecolor(tuple(header_color))

        for _ in leads:
            row_idx += 1
            label_cell = table[row_idx, -1]
            label_cell.set_text_props(fontsize=9)
            if model == "hres":
                label_cell.set_facecolor(tuple(hres_grey))

    ax.set_title(
        "Physics Metrics Summary — By Model\n"
        "(White = 0, Blue = positive, Red = negative, *** = nearest available lead time)",
        fontsize=13, fontweight="bold", pad=12,
    )
    out = outdir / "summary_table_combined_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--ifs", action="store_true",
                        help="Use IFS HRES comparison files (*_ifs.csv) instead of ERA5")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(__file__).resolve().parent.parent / "results"
    )
    outdir = results_dir / ("plots_combined_IFS" if args.ifs else "plots_combined")
    outdir.mkdir(exist_ok=True)

    summaries = load_summaries(results_dir, ifs_mode=args.ifs)
    if not summaries:
        print("No summary CSVs found.")
        return

    # Lead times: 12h, 5d, 10d (aligned with NeuralGCM)
    leads = [12, 120, 240]
    
    # Generate both table layouts
    render_combined_table(leads, summaries, outdir)
    render_combined_table_by_model(leads, summaries, outdir)


if __name__ == "__main__":
    main()
