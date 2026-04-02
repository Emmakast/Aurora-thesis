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

MODELS = ["hres", "fuxi", "graphcast", "neuralgcm", "pangu", "aurora"]
NICE = {
    "hres": "HRES", "fuxi": "FuXi", "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM", "pangu": "Pangu", "aurora": "Aurora",
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

# Split metrics into groups for separate tables
METRICS_RMSE_CONSERVATION = [
    "hydrostatic_rmse", "geostrophic_rmse",
    "dry_mass_drift_pct_per_day", "water_mass_drift_pct_per_day",
    "total_energy_drift_pct_per_day",
]
METRICS_SPECTRAL = [
    "effective_resolution_km", "spectral_residual",
    "spectral_divergence", "small_scale_ratio",
]

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
        # For HRES in non-IFS mode, use the _vs_era5 file if it exists
        if m == "hres" and not ifs_mode:
            p = results_dir / f"physics_summary_{m}_2020_vs_era5.csv"
            if not p.exists():
                p = results_dir / f"physics_summary_{m}_2020.csv"
        elif m == "aurora":
            p = results_dir / f"physics_summary_aurora_s3_2022{suffix}.csv"
        else:
            p = results_dir / f"physics_summary_{m}_2020{suffix}.csv"
        if p.exists():
            out[m] = pd.read_csv(p)
            print(f"Loaded {p.name}")
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
    out = outdir / "summary_table_combined_aurora.png"
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
        out2 = outdir / f"summary_table_{lead_str}_aurora.png"
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
    out = outdir / "summary_table_combined_model_aurora.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")


def render_split_tables(leads: list[int], summaries: dict[str, pd.DataFrame],
                        outdir: Path):
    """Render two separate tables: RMSE+Conservation and Spectral metrics."""
    hres_grey = np.array([0.94, 0.94, 0.94])
    header_color = np.array([0.82, 0.82, 0.82])
    model_labels = [NICE.get(m, m) for m in MODELS]

    for metrics_list, title_suffix, filename_suffix in [
        (METRICS_RMSE_CONSERVATION, "RMSE & Conservation", "rmse_conservation"),
        (METRICS_SPECTRAL, "Spectral", "spectral"),
    ]:
        n_metrics = len(metrics_list)
        n_models = len(MODELS)
        col_labels = [METRICS[m][0] for m in metrics_list]

        n_leads = len(leads)
        total_rows = n_leads * (n_models + 1)

        fig_w = max(2.0 * n_metrics, 12)
        fig_h = 0.48 * total_rows + 1.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        all_texts = []
        all_colors = []
        all_row_labels = []

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
            row_idx += 1
            for j in range(n_metrics):
                table[row_idx, j].set_facecolor(tuple(header_color))
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
            f"Physics Metrics Summary — {title_suffix}\n"
            "(White = 0, Blue = positive, Red = negative, *** = nearest available lead time)",
            fontsize=13, fontweight="bold", pad=12,
        )
        out = outdir / f"summary_table_{filename_suffix}_aurora.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved {out}")


def render_unified_png_table(leads: list[int], summaries: dict[str, pd.DataFrame], outdir: Path):
    """Renders a single unified PNG table with metrics on rows and models on columns."""
    
    models_to_plot = ["hres", "pangu", "graphcast", "fuxi", "neuralgcm", "aurora"]
    model_labels = [NICE.get(m, m) for m in models_to_plot]

    categories = {
        "Conservation": [
            ("dry_mass_drift_pct_per_day", "Dry Mass (→ 0)"),
            ("water_mass_drift_pct_per_day", "Water Mass (→ 0)"),
            ("total_energy_drift_pct_per_day", "Total Energy (→ 0)")
        ],
        "Structural": [
            ("effective_resolution_km", "Eff. Res. km (↓)"),
            ("spectral_divergence", "Spec. Div. (↓)"),
            ("spectral_residual", "Spec. Res. (↓)")
        ],
        "Dynamical": [
            ("geostrophic_rmse", "Geostrophic (↓)"),
            ("hydrostatic_rmse", "Hydrostatic (↓)")
        ]
    }

    # 1. Calculate max abs per metric for color scaling
    max_abs = {}
    for cat, metrics in categories.items():
        for m_key, _ in metrics:
            vals = []
            for m in models_to_plot:
                if m in summaries:
                    df = summaries[m]
                    for lead in leads:
                        df_lt = df[df["lead_time_hours"] == lead]
                        if df_lt.empty:
                            avail = sorted(df["lead_time_hours"].unique())
                            nearest = min(avail, key=lambda x: abs(x - lead))
                            df_lt = df[df["lead_time_hours"] == nearest]
                        val = get_value(df_lt, m_key)
                        if not np.isnan(val):
                            vals.append(abs(val))
            max_abs[m_key] = max(vals) if vals else 1.0
            if max_abs[m_key] == 0: max_abs[m_key] = 1.0

    # 2. Build table data
    cell_texts = []
    cell_colors = []

    white = np.array([1.0, 1.0, 1.0])
    blue = np.array([0.7, 0.85, 1.0])
    red = np.array([1.0, 0.75, 0.75])
    header_color = np.array([0.9, 0.9, 0.9])
    cat_color = np.array([0.95, 0.95, 0.95])

    # Header Row 1: Lead times
    row0_text = ["", ""]
    row0_color = [header_color, header_color]
    for lead in leads:
        block = [""] * len(models_to_plot)
        block[len(models_to_plot)//2] = f"Lead Time: {lead}h"
        row0_text.extend(block)
        row0_color.extend([header_color] * len(models_to_plot))
    cell_texts.append(row0_text)
    cell_colors.append(row0_color)

    # Header Row 2: Models
    row1_text = ["Category", "Metric"] + model_labels * len(leads)
    row1_color = [header_color, header_color] + [header_color] * (len(models_to_plot) * len(leads))
    cell_texts.append(row1_text)
    cell_colors.append(row1_color)

    # Data Rows
    for cat_name, metrics in categories.items():
        for i, (m_key, m_label) in enumerate(metrics):
            row_t = [cat_name if i == len(metrics)//2 else "", m_label]
            row_c = [cat_color, cat_color]

            for lead in leads:
                for m in models_to_plot:
                    if m not in summaries:
                        row_t.append("—")
                        row_c.append(white)
                        continue

                    df = summaries[m]
                    df_lt = df[df["lead_time_hours"] == lead]
                    is_appx = False
                    if df_lt.empty:
                        avail = sorted(df["lead_time_hours"].unique())
                        nearest = min(avail, key=lambda x: abs(x - lead))
                        df_lt = df[df["lead_time_hours"] == nearest]
                        is_appx = True

                    val = get_value(df_lt, m_key)
                    if np.isnan(val):
                        row_t.append("—")
                        row_c.append(white)
                    else:
                        text = fmt(val, m_key)
                        if is_appx: text += "*"
                        row_t.append(text)

                        # Apply color logic
                        intensity = min(abs(val) / max_abs[m_key], 1.0) * 0.8
                        if val > 0: c = white * (1 - intensity) + blue * intensity
                        elif val < 0: c = white * (1 - intensity) + red * intensity
                        else: c = white
                        row_c.append(c)

            cell_texts.append(row_t)
            cell_colors.append(row_c)

    # 3. Matplotlib rendering
    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)
    fig_w = max(1.2 * n_cols, 16)
    fig_h = max(0.4 * n_rows, 6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    colWidths = [0.08, 0.17] + [0.05] * (len(leads) * len(models_to_plot))
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=colWidths,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Styling headers
    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold", fontsize=11)
        table[1, j].set_text_props(fontweight="bold", fontsize=10)

    # Styling category and metric names
    for i in range(2, n_rows):
        table[i, 0].set_text_props(fontweight="bold")
        table[i, 1].set_text_props(fontweight="bold")

    # --- Cell Merging ---
    # Top-left empty 2x2 area
    table[0, 0].visible_edges = ''
    table[0, 1].visible_edges = ''

    # Top row Lead times merging
    for l_idx in range(len(leads)):
        start_col = 2 + l_idx * len(models_to_plot)
        end_col = start_col + len(models_to_plot) - 1
        for c in range(start_col, end_col + 1):
            if c == start_col:
                table[0, c].visible_edges = 'LBT'
            elif c == end_col:
                table[0, c].visible_edges = 'RBT'
            else:
                table[0, c].visible_edges = 'BT'

    # First Column (Categories) merging
    row_idx = 2
    for _, metrics in categories.items():
        start_row = row_idx
        end_row = row_idx + len(metrics) - 1
        for r in range(start_row, end_row + 1):
            if r == start_row:
                table[r, 0].visible_edges = 'LRT'
            elif r == end_row:
                table[r, 0].visible_edges = 'LRB'
            else:
                table[r, 0].visible_edges = 'LR'
        row_idx += len(metrics)

    # --- Draw Thick Lines ---
    # Fix axes limits so that drawing plot lines doesn't move the table
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.autoscale(False)
    fig.canvas.draw()

    x_left = table[0, 0].get_x()
    x_metric = table[0, 1].get_x() + table[0, 1].get_width()
    x_right = table[0, n_cols - 1].get_x() + table[0, n_cols - 1].get_width()

    y_top = table[0, 0].get_y() + table[0, 0].get_height()
    y_bot = table[n_rows - 1, 0].get_y()
    y_head_row0 = table[0, 0].get_y()
    y_head_row1 = table[1, 0].get_y()

    # Outer Border
    ax.plot([x_left, x_right], [y_top, y_top], color='black', linewidth=2.5, clip_on=False)
    ax.plot([x_left, x_right], [y_bot, y_bot], color='black', linewidth=2.5, clip_on=False)
    ax.plot([x_left, x_left], [y_bot, y_top], color='black', linewidth=2.5, clip_on=False)
    ax.plot([x_right, x_right], [y_bot, y_top], color='black', linewidth=2.5, clip_on=False)

    # Header borders
    ax.plot([x_left, x_right], [y_head_row1, y_head_row1], color='black', linewidth=2.5, clip_on=False)
    ax.plot([x_metric, x_right], [y_head_row0, y_head_row0], color='black', linewidth=2.5, clip_on=False)

    # Vertical Categories separator
    ax.plot([x_metric, x_metric], [y_bot, y_top], color='black', linewidth=2.5, clip_on=False)
    x_cat_edge = table[0, 0].get_x() + table[0, 0].get_width()
    ax.plot([x_cat_edge, x_cat_edge], [y_bot, y_head_row1], color='black', linewidth=2.5, clip_on=False)

    # Lead time separators
    for l_idx in range(1, len(leads)):
        c = 2 + l_idx * len(models_to_plot)
        x = table[0, c].get_x()
        ax.plot([x, x], [y_bot, y_top], color='black', linewidth=2.5, clip_on=False)

    # Category separators
    row_bounds = [
        1 + len(categories["Conservation"]),
        1 + len(categories["Conservation"]) + len(categories["Structural"]),
    ]
    for end_row in row_bounds:
        y = table[end_row, 0].get_y()
        ax.plot([x_left, x_right], [y, y], color='black', linewidth=2.5, clip_on=False)

    ax.set_title("Unified Physics Metrics Summary\n(White=0, Blue=Pos Drift, Red=Neg Drift, *=Nearest Lead)", fontweight="bold", fontsize=14, pad=20)

    out = outdir / "summary_table_unified_transposed_aurora.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
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
    
    # Generate all table layouts
    render_combined_table(leads, summaries, outdir)
    render_combined_table_by_model(leads, summaries, outdir)
    render_split_tables(leads, summaries, outdir)
    render_unified_png_table(leads, summaries, outdir)


if __name__ == "__main__":
    main()
