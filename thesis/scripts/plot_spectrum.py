#!/usr/bin/env python
"""
Unified spectral plotting — KE (500 hPa), KE (850 hPa), or Q spectrum.

Produces for the chosen analysis:
  1. Per-model spectrum (wavenumber x-axis)
  2. Per-model spectrum (wavelength x-axis)
  3. Combined multi-model spectrum at key lead times (both axes)
  4. Spectral ratio E_pred/E_ERA5 per model (wavelength x-axis)
  5. Combined spectral ratio at key lead times
  6. Sensitivity table: effective resolution at multiple thresholds

Usage:
    python plot_spectrum.py ke                         # KE 500 hPa
    python plot_spectrum.py ke_850hpa                  # KE 850 hPa
    python plot_spectrum.py q                          # Specific humidity
    python plot_spectrum.py ke --models pangu graphcast
    python plot_spectrum.py ke --thresholds 0.3 0.5 0.7 0.9
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EARTH_RADIUS_KM = 6371.0

MODEL_STYLES = {
    "aurora":    {"color": "#1f77b4", "label": "Aurora"},
    "pangu":     {"color": "#ff7f0e", "label": "Pangu-Weather"},
    "fuxi":      {"color": "#2ca02c", "label": "FuXi"},
    "graphcast": {"color": "#d62728", "label": "GraphCast"},
    "neuralgcm": {"color": "#9467bd", "label": "NeuralGCM"},
    "hres":      {"color": "#8c564b", "label": "HRES"},
}

LEAD_CMAP = plt.cm.viridis_r
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


# ── Analysis configuration ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisCfg:
    """All the strings that differ between ke / ke_850hpa / q."""
    csv_prefix: str          # e.g. "ke_spectrum_850hpa"
    value_col: str           # "energy" or "power"
    wide_pred: str           # "energy_pred" or "power_pred"
    wide_era5: str           # "energy_era5" or "power_era5"
    ylabel: str              # y-axis label (LaTeX ok)
    title: str               # title prefix, e.g. "KE Spectrum (850 hPa)"
    ratio_ylabel: str        # y-axis on ratio plots
    outdir_name: str         # output sub-directory name


ANALYSES: dict[str, AnalysisCfg] = {
    "ke": AnalysisCfg(
        csv_prefix="ke_spectrum",
        value_col="energy",
        wide_pred="energy_pred",
        wide_era5="energy_era5",
        ylabel=r"Kinetic Energy $E(l)$",
        title="KE Spectrum",
        ratio_ylabel=r"$E_{pred}(l) \;/\; E_{ERA5}(l)$",
        outdir_name="plots_ke_spectrum",
    ),
    "ke_850hpa": AnalysisCfg(
        csv_prefix="ke_spectrum_850hpa",
        value_col="energy",
        wide_pred="energy_pred",
        wide_era5="energy_era5",
        ylabel=r"Kinetic Energy $E(l)$",
        title="KE Spectrum (850 hPa)",
        ratio_ylabel=r"$E_{pred}(l) \;/\; E_{ERA5}(l)$",
        outdir_name="plots_ke_spectrum_850hpa",
    ),
    "q": AnalysisCfg(
        csv_prefix="q_spectrum",
        value_col="power",
        wide_pred="power_pred",
        wide_era5="power_era5",
        ylabel=r"Specific Humidity Power $S_q(l)$",
        title="Humidity Spectrum",
        ratio_ylabel=r"$S_{pred}(l) \;/\; S_{ERA5}(l)$",
        outdir_name="plots_q_spectrum",
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wl(l: np.ndarray) -> np.ndarray:
    """Spherical harmonic degree → wavelength (km)."""
    with np.errstate(divide="ignore"):
        return 2.0 * np.pi * EARTH_RADIUS_KM / l


def _lead_label(lh: int) -> str:
    return f"{lh}h" if lh < 24 else f"Day {lh // 24}"


def _model_style(model: str) -> dict:
    return MODEL_STYLES.get(model, {"color": "grey", "label": model})


def _extract_model(path: str, prefix: str) -> str:
    """Strip prefix and trailing _YYYY or _YYYY_ifs from stem → model name."""
    stem = Path(path).stem.replace(f"{prefix}_", "")
    # Handle _ifs suffix: ke_spectrum_ifs_fuxi_2020 → fuxi
    if stem.endswith("_ifs"):
        stem = stem[:-4]  # Remove _ifs
    parts = stem.rsplit("_", 1)
    return parts[0] if (len(parts) == 2 and parts[1].isdigit()) else stem


# ── Data loading ─────────────────────────────────────────────────────────────

def _to_long(df: pd.DataFrame, model: str, cfg: AnalysisCfg) -> pd.DataFrame:
    """Normalise CSV (long or wide format) to long with model/source columns."""
    vcol = cfg.value_col
    if "source" in df.columns and vcol in df.columns:
        if "model" not in df.columns:
            df = df.copy()
            df["model"] = model
        return df
    # Wide format
    pred = df[["date", "lead_hours", "wavenumber", cfg.wide_pred]].copy()
    pred["source"] = "pred"
    pred = pred.rename(columns={cfg.wide_pred: vcol})
    era5 = df[["date", "lead_hours", "wavenumber", cfg.wide_era5]].copy()
    era5["source"] = "era5"
    era5 = era5.rename(columns={cfg.wide_era5: vcol})
    long = pd.concat([pred, era5], ignore_index=True)
    long["model"] = model
    return long


def _normalise_wide(df: pd.DataFrame, cfg: AnalysisCfg) -> pd.DataFrame:
    """Ensure wide format with pred/era5 columns."""
    if cfg.wide_pred in df.columns and cfg.wide_era5 in df.columns:
        return df
    if "source" in df.columns and cfg.value_col in df.columns:
        wide = df.pivot_table(
            index=["date", "lead_hours", "wavenumber"],
            columns="source",
            values=cfg.value_col,
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(columns={"pred": cfg.wide_pred, "era5": cfg.wide_era5})
        return wide
    raise ValueError(f"Unrecognised columns: {list(df.columns)}")


def load_spectra_long(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
    ifs_mode: bool = False,
) -> pd.DataFrame:
    """Load CSVs into long format (for wavenumber-axis plots)."""
    suffix = "_ifs" if ifs_mode else ""
    pattern = str(results_dir / f"{cfg.csv_prefix}_*{suffix}.csv")
    csvs = sorted(glob.glob(pattern))
    # For 'ke' analysis, exclude 850hpa files that also match the glob
    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]
    # In IFS mode, only keep files ending with _ifs.csv
    if ifs_mode:
        csvs = [p for p in csvs if Path(p).stem.endswith("_ifs")]
    else:
        csvs = [p for p in csvs if not Path(p).stem.endswith("_ifs")]
    if not csvs:
        raise FileNotFoundError(f"No {cfg.csv_prefix}_*{suffix}.csv found in {results_dir}")

    exclude = exclude or set()
    dfs = []
    for p in csvs:
        model = _extract_model(p, cfg.csv_prefix)
        if model in exclude or (models and model not in models):
            continue
        df = _to_long(pd.read_csv(p), model, cfg)
        if df[df["source"] == "pred"].empty:
            continue
        dfs.append(df)
        print(f"  Loaded {Path(p).name}: {len(df):,} rows  (model={model})")
    if not dfs:
        raise FileNotFoundError(f"No matching CSVs after filtering in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def load_spectra_wide(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
    ifs_mode: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load CSVs into wide format, averaged over dates (for wavelength/ratio plots)."""
    suffix = "_ifs" if ifs_mode else ""
    pattern = str(results_dir / f"{cfg.csv_prefix}_*{suffix}.csv")
    csvs = sorted(glob.glob(pattern))
    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]
    # In IFS mode, only keep files ending with _ifs.csv
    if ifs_mode:
        csvs = [p for p in csvs if Path(p).stem.endswith("_ifs")]
    else:
        csvs = [p for p in csvs if not Path(p).stem.endswith("_ifs")]
    if not csvs:
        raise FileNotFoundError(f"No {cfg.csv_prefix}_*.csv found in {results_dir}")

    exclude = exclude or set()
    out: dict[str, pd.DataFrame] = {}
    for p in csvs:
        model = _extract_model(p, cfg.csv_prefix)
        if model in exclude or (models and model not in models):
            continue
        df = _normalise_wide(pd.read_csv(p), cfg)
        agg = (
            df.groupby(["lead_hours", "wavenumber"])[[cfg.wide_pred, cfg.wide_era5]]
            .mean()
            .reset_index()
        )
        out[model] = agg
        print(f"  Loaded {Path(p).name}: {len(df):,} → {len(agg):,} after averaging  (model={model})")
    return out


def _mean_spectrum(df: pd.DataFrame, vcol: str) -> pd.DataFrame:
    return (
        df.groupby(["model", "lead_hours", "wavenumber", "source"], as_index=False)
        [vcol].mean()
    )


# ── Spectrum plots (wavenumber x-axis) ───────────────────────────────────────

def plot_per_model_wn(df: pd.DataFrame, outdir: Path, cfg: AnalysisCfg, ref_label: str = "ERA5"):
    """Per-model: spectrum at each lead time + reference (wavenumber x-axis)."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)
    vcol = cfg.value_col

    for model, mdf in df.groupby("model"):
        lead_times = sorted(mdf["lead_hours"].unique())
        colors = LEAD_CMAP(np.linspace(0, 0.85, len(lead_times)))

        fig, ax = plt.subplots(figsize=(12, 7))
        era5_df = mdf[(mdf["source"] == "era5") & (mdf["lead_hours"] == lead_times[0])]
        if not era5_df.empty:
            ax.loglog(era5_df["wavenumber"], era5_df[vcol],
                      color="black", linewidth=2, alpha=0.7, label=ref_label, zorder=5)

        for i, lh in enumerate(lead_times):
            pred = mdf[(mdf["source"] == "pred") & (mdf["lead_hours"] == lh)]
            if pred.empty:
                continue
            ax.loglog(pred["wavenumber"], pred[vcol],
                      color=colors[i], linewidth=1.5, alpha=0.85, label=_lead_label(lh))

        ax.set_title(f"{cfg.title} — {_model_style(model)['label']}", fontsize=14, pad=12)
        ax.set_xlabel("Spherical harmonic degree $l$", fontsize=12)
        ax.set_ylabel(cfg.ylabel, fontsize=12)
        ax.set_xlim(1, None)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / f"{cfg.csv_prefix}_{model}.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {cfg.csv_prefix}_{model}.png")


def plot_combined_wn(df: pd.DataFrame, outdir: Path, cfg: AnalysisCfg, target_lead: int, ref_label: str = "ERA5"):
    """All models at one lead time (wavenumber x-axis)."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)
    vcol = cfg.value_col

    fig, ax = plt.subplots(figsize=(12, 7))
    era5_df = df[(df["source"] == "era5") & (df["lead_hours"] == target_lead)]
    if not era5_df.empty:
        era5_mean = era5_df.groupby("wavenumber")[vcol].mean()
        ax.loglog(era5_mean.index, era5_mean.values,
                  color="black", linewidth=2, alpha=0.7, label=ref_label, zorder=5)

    for model in sorted(df["model"].unique()):
        pred = df[(df["model"] == model) & (df["source"] == "pred") & (df["lead_hours"] == target_lead)]
        if pred.empty:
            continue
        s = _model_style(model)
        ax.loglog(pred["wavenumber"], pred[vcol],
                  color=s["color"], linewidth=1.5, alpha=0.85, label=s["label"])

    ax.set_title(f"{cfg.title} Comparison — {_lead_label(target_lead)}", fontsize=14, pad=12)
    ax.set_xlabel("Spherical harmonic degree $l$", fontsize=12)
    ax.set_ylabel(cfg.ylabel, fontsize=12)
    ax.set_xlim(1, None)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fname = f"{cfg.csv_prefix}_combined_{target_lead}h.png"
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Spectrum plots (wavelength x-axis) ───────────────────────────────────────

def plot_per_model_wl(models: dict[str, pd.DataFrame], outdir: Path, cfg: AnalysisCfg, ref_label: str = "ERA5"):
    """Per-model: spectrum at each lead time + reference (wavelength x-axis)."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    for model, df in sorted(models.items()):
        lead_times = sorted(df["lead_hours"].unique())
        colors = LEAD_CMAP(np.linspace(0, 0.85, len(lead_times)))

        fig, ax = plt.subplots(figsize=(12, 7))

        era5 = df[(df["lead_hours"] == lead_times[0]) & (df["wavenumber"] > 0)]
        if not era5.empty:
            ax.loglog(_wl(era5["wavenumber"].values), era5[cfg.wide_era5].values,
                      color="black", linewidth=2, alpha=0.7, label=ref_label, zorder=5)

        for i, lh in enumerate(lead_times):
            sub = df[(df["lead_hours"] == lh) & (df["wavenumber"] > 0)]
            ax.loglog(_wl(sub["wavenumber"].values), sub[cfg.wide_pred].values,
                      color=colors[i], linewidth=1.5, alpha=0.85, label=_lead_label(lh))

        ax.set_title(f"{cfg.title} — {_model_style(model)['label']}", fontsize=14, pad=12)
        ax.set_xlabel("Wavelength (km)", fontsize=12)
        ax.set_ylabel(cfg.ylabel, fontsize=12)
        ax.invert_xaxis()
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        fname = f"{cfg.csv_prefix}_wavelength_{model}.png"
        fig.savefig(outdir / fname, dpi=300)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_combined_wl(
    models: dict[str, pd.DataFrame], outdir: Path, cfg: AnalysisCfg, target_lead: int, ref_label: str = "ERA5",
):
    """All models at one lead time (wavelength x-axis)."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    first = next(iter(models.values()))
    era5 = first[(first["lead_hours"] == target_lead) & (first["wavenumber"] > 0)]
    if not era5.empty:
        ax.loglog(_wl(era5["wavenumber"].values), era5[cfg.wide_era5].values,
                  color="black", linewidth=2, alpha=0.7, label=ref_label, zorder=5)

    for model, df in sorted(models.items()):
        sub = df[(df["lead_hours"] == target_lead) & (df["wavenumber"] > 0)]
        if sub.empty:
            continue
        s = _model_style(model)
        ax.loglog(_wl(sub["wavenumber"].values), sub[cfg.wide_pred].values,
                  color=s["color"], linewidth=1.5, alpha=0.85, label=s["label"])

    ax.set_title(f"{cfg.title} Comparison — {_lead_label(target_lead)}", fontsize=14, pad=12)
    ax.set_xlabel("Wavelength (km)", fontsize=12)
    ax.set_ylabel(cfg.ylabel, fontsize=12)
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    fig.tight_layout()
    fname = f"{cfg.csv_prefix}_wavelength_combined_{target_lead}h.png"
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Ratio plots (wavelength x-axis) ─────────────────────────────────────────

def plot_ratio_per_model(
    models: dict[str, pd.DataFrame], outdir: Path, cfg: AnalysisCfg,
    thresholds: list[float],
):
    """Per-model: E_pred/E_ERA5 vs wavelength, one line per lead time."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    for model, df in sorted(models.items()):
        lead_times = sorted(df["lead_hours"].unique())
        colors = LEAD_CMAP(np.linspace(0, 0.85, len(lead_times)))

        fig, ax = plt.subplots(figsize=(12, 7))

        for i, lh in enumerate(lead_times):
            sub = df[(df["lead_hours"] == lh) & (df["wavenumber"] > 0)].copy()
            sub = sub[sub[cfg.wide_era5] > 1e-12]
            wl = _wl(sub["wavenumber"].values)
            ratio = sub[cfg.wide_pred].values / sub[cfg.wide_era5].values
            ax.semilogx(wl, ratio, color=colors[i], linewidth=1.2, alpha=0.85,
                        label=_lead_label(lh))

        for thr in thresholds:
            ax.axhline(y=thr, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(ax.get_xlim()[0] * 1.05, thr + 0.015, f"{thr}",
                    fontsize=8, color="grey", va="bottom")
        ax.axhline(y=1.0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)

        ax.set_title(
            f"Spectral Ratio — {_model_style(model)['label']}",
            fontsize=14, pad=12,
        )
        ax.set_xlabel("Wavelength (km)", fontsize=12)
        ax.set_ylabel(cfg.ratio_ylabel, fontsize=12)
        ax.set_ylim(-0.05, None)
        ax.invert_xaxis()
        ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()
        fname = f"{cfg.csv_prefix}_ratio_{model}.png"
        fig.savefig(outdir / fname, dpi=300)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_ratio_combined(
    models: dict[str, pd.DataFrame], outdir: Path, cfg: AnalysisCfg,
    thresholds: list[float], target_lead: int,
):
    """All models at one lead time — spectral ratio."""
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    for model, df in sorted(models.items()):
        sub = df[(df["lead_hours"] == target_lead) & (df["wavenumber"] > 0)].copy()
        if sub.empty:
            continue
        sub = sub[sub[cfg.wide_era5] > 1e-12]
        wl = _wl(sub["wavenumber"].values)
        ratio = sub[cfg.wide_pred].values / sub[cfg.wide_era5].values
        s = _model_style(model)
        ax.semilogx(wl, ratio, color=s["color"], linewidth=1.5, alpha=0.85,
                    label=s["label"])

    for thr in thresholds:
        ax.axhline(y=thr, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(ax.get_xlim()[0] * 1.05, thr + 0.015, f"{thr}",
                fontsize=8, color="grey", va="bottom")
    ax.axhline(y=1.0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)

    ax.set_title(
        f"Spectral Ratio — All Models — {_lead_label(target_lead)}", fontsize=14, pad=12,
    )
    ax.set_xlabel("Wavelength (km)", fontsize=12)
    ax.set_ylabel(cfg.ratio_ylabel, fontsize=12)
    ax.set_ylim(-0.05, None)
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    fig.tight_layout()
    fname = f"{cfg.csv_prefix}_ratio_combined_{target_lead}h.png"
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Sensitivity table ────────────────────────────────────────────────────────

def _find_effective_resolution(
    wavenumber: np.ndarray,
    ratio: np.ndarray,
    threshold: float,
    k_min: int = 10,
    n_consecutive: int = 5,
) -> float:
    """Effective resolution (km) at a given ratio threshold."""
    mask = wavenumber >= k_min
    k_sel, r_sel = wavenumber[mask], ratio[mask]
    below = r_sel < threshold
    n = len(r_sel)
    fallback = 2.0 * np.pi * EARTH_RADIUS_KM / (float(k_sel[-1]) if len(k_sel) else float(wavenumber[-1]))

    if n < n_consecutive:
        return fallback
    run, idx = 0, None
    for i in range(n):
        if below[i]:
            run += 1
            if run >= n_consecutive:
                idx = i - n_consecutive + 1
                break
        else:
            run = 0
    if idx is None:
        return fallback
    return 2.0 * np.pi * EARTH_RADIUS_KM / float(k_sel[idx])


# ---------------------------------------------------------------------------
# Effective Resolution: Order-of-Operations
# ---------------------------------------------------------------------------
# L_eff is computed from the MEAN spectrum across all init dates, NOT by
# averaging per-date L_eff values.  This is a deliberate methodological
# choice:
#
#   Correct:   L_eff = f( <E(k)>_dates )      — drop-off on the mean curve
#   Wrong:     L_eff = < f(E_date(k)) >_dates  — mean of per-date drop-offs
#
# The per-date approach is sensitive to daily noise in the spectrum tails,
# which can trigger premature threshold crossings and systematically
# underestimate L_eff.  By averaging the spectra first, we obtain a smooth
# climatological curve whose drop-off point is physically meaningful.
# ---------------------------------------------------------------------------


def sensitivity_table(
    models: dict[str, pd.DataFrame], cfg: AnalysisCfg,
    thresholds: list[float], outdir: Path,
) -> pd.DataFrame:
    rows = []
    for model, df in sorted(models.items()):
        for lh in sorted(df["lead_hours"].unique()):
            sub = df[(df["lead_hours"] == lh) & (df["wavenumber"] > 0)].copy()
            sub = sub[sub[cfg.wide_era5] > 1e-12]
            wn = sub["wavenumber"].values
            ratio = sub[cfg.wide_pred].values / sub[cfg.wide_era5].values

            row: dict = {"model": model, "lead_hours": int(lh)}
            for thr in thresholds:
                row[f"eff_res_{thr}"] = round(
                    _find_effective_resolution(wn, ratio, thr), 1
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"sensitivity_effective_resolution_{cfg.csv_prefix}.csv"
    result.to_csv(csv_path, index=False)
    print(f"\n  Saved sensitivity table → {csv_path}")

    # Pretty-print
    thr_cols = [c for c in result.columns if c.startswith("eff_res_")]
    header = f"  {'Model':<12} {'Lead':>6}"
    for c in thr_cols:
        header += f"  thr={c.replace('eff_res_', ''):>4}"
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))
    for _, r in result.iterrows():
        line = f"  {r['model']:<12} {int(r['lead_hours']):>5}h"
        for c in thr_cols:
            line += f"  {r[c]:>9.1f}"
        print(line)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified spectral plotting for KE / KE-850 / Q spectra."
    )
    parser.add_argument(
        "analysis", choices=list(ANALYSES),
        help="Analysis type: ke (500 hPa), ke_850hpa, or q",
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific model names")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Model names to exclude")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--combined-lead", type=int, default=240,
                        help="Primary lead time for combined plots (default: 240)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                        help="Ratio thresholds for sensitivity table (default: 0.3 0.5 0.7 0.9)")
    parser.add_argument("--ifs", action="store_true",
                        help="Use IFS HRES t=0 comparison files (*_ifs.csv)")
    args = parser.parse_args()

    cfg = ANALYSES[args.analysis]
    results_dir = Path(args.results_dir)
    exclude = set(args.exclude)
    thresholds = sorted(args.thresholds)
    ifs_mode = args.ifs

    # Modify config for IFS mode (titles and output directory only)
    if ifs_mode:
        cfg = AnalysisCfg(
            csv_prefix=cfg.csv_prefix,  # Keep original prefix for file matching
            value_col=cfg.value_col,
            wide_pred=cfg.wide_pred,
            wide_era5=cfg.wide_era5,
            ylabel=cfg.ylabel,
            title=cfg.title + " (vs IFS HRES)",
            ratio_ylabel=cfg.ratio_ylabel.replace("ERA5", "IFS"),
            outdir_name=cfg.outdir_name + "_ifs",
        )

    print(f"=== {cfg.title} ===")
    print(f"Loading from {results_dir} ...\n")

    # Load both representations
    df_long = load_spectra_long(results_dir, cfg, models=args.models, exclude=exclude, ifs_mode=ifs_mode)
    models_wide = load_spectra_wide(results_dir, cfg, models=args.models, exclude=exclude, ifs_mode=ifs_mode)

    df_mean = _mean_spectrum(df_long, cfg.value_col)
    print(f"\nTotal: {len(df_mean):,} rows after averaging ({df_mean['model'].nunique()} models)")

    outdir = results_dir / cfg.outdir_name

    # Collect all available lead times
    all_leads = sorted({int(lh) for df in models_wide.values() for lh in df["lead_hours"].unique()})
    key_leads = [lt for lt in [12, 120, 240, args.combined_lead] if lt in all_leads]
    # Deduplicate while preserving order
    key_leads = list(dict.fromkeys(key_leads))

    # Reference label for plots
    ref_label = "IFS HRES" if ifs_mode else "ERA5"

    # 1. Per-model spectrum (wavenumber)
    print("\n--- Per-model spectrum (wavenumber) ---")
    plot_per_model_wn(df_mean, outdir, cfg, ref_label=ref_label)

    # 2. Combined spectrum (wavenumber)
    print("\n--- Combined spectrum (wavenumber) ---")
    for lt in key_leads:
        plot_combined_wn(df_mean, outdir, cfg, target_lead=lt, ref_label=ref_label)

    # 3. Per-model spectrum (wavelength)
    print("\n--- Per-model spectrum (wavelength) ---")
    plot_per_model_wl(models_wide, outdir, cfg, ref_label=ref_label)

    # 4. Combined spectrum (wavelength)
    print("\n--- Combined spectrum (wavelength) ---")
    for lt in key_leads:
        plot_combined_wl(models_wide, outdir, cfg, target_lead=lt, ref_label=ref_label)

    # 5. Spectral ratio per model
    print("\n--- Per-model spectral ratio ---")
    plot_ratio_per_model(models_wide, outdir, cfg, thresholds)

    # 6. Combined spectral ratio
    print("\n--- Combined spectral ratio ---")
    for lt in all_leads:
        plot_ratio_combined(models_wide, outdir, cfg, thresholds, target_lead=lt)

    # 7. Sensitivity table
    print("\n--- Sensitivity table ---")
    sensitivity_table(models_wide, cfg, thresholds, outdir)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
