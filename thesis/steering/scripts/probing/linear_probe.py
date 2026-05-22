#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Linear Probe Steering for Aurora

Trains a ridge regression (linear probe) to predict a continuous climate index
(AO, AAO, ENSO) directly from Aurora encoder latent representations.

The learned weight matrix W is then used as a continuous steering vector:
injecting α·W during inference "slides" the latent along the primary axis
of the targeted physical process.

Pipeline:
  1. Load latent tensors + ground-truth index values
  2. Flatten latents → feature matrix X, index → target vector y
  3. Train Ridge regression: W = argmin ||XW - y||² + λ||W||²
  4. Reshape W back to latent geometry → steering vector
  5. (Optional) Run steered Aurora inference with the probe vector
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import boto3
    from dotenv import load_dotenv
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Add the steering script directory to path to import helpers
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/scripts/steering')
try:
    from extract_latents_hres import prepare_batch, batch_to_dataset, download_data, download_static
except ImportError:
    print("Warning: Could not import helpers from extract_latents_hres.py")

try:
    from aurora import Aurora, rollout
except ImportError:
    print("Warning: Could not import Aurora. Make sure the aurora environment is active.")


# ══════════════════════════════════════════════════════════════════════════════
# Index Loaders
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/data")

def load_ao_index() -> pd.DataFrame:
    """Load daily AO index (CDAS z1000)."""
    df = pd.read_csv(DATA_DIR / "norm.daily.ao.cdas.z1000.19500101_current.csv")
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(columns={"year": "year", "month": "month", "day": "day"})
    )
    df = df.rename(columns={"ao_index_cdas": "index_value"})
    return df[["date", "index_value"]]


def load_aao_index() -> pd.DataFrame:
    """Load daily AAO index (CDAS z700)."""
    df = pd.read_csv(DATA_DIR / "norm.daily.aao.cdas.z700.19790101_current.csv")
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(columns={"year": "year", "month": "month", "day": "day"})
    )
    df = df.rename(columns={"aao_index_cdas": "index_value"})
    return df[["date", "index_value"]]


def load_enso_index() -> pd.DataFrame:
    """Load monthly SOI index and expand to daily resolution."""
    df = pd.read_csv(
        DATA_DIR / "soi.long.csv",
        skiprows=1,  # skip the header description line
        names=["date_str", "index_value"],
        skipinitialspace=True,
    )
    df["date"] = pd.to_datetime(df["date_str"])
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
    # Drop missing values (-99.99)
    df = df[df["index_value"] > -90]
    # Expand monthly → daily via forward fill
    df = df.set_index("date").resample("D").ffill().reset_index()
    return df[["date", "index_value"]]


INDEX_LOADERS = {
    "AO": load_ao_index,
    "AAO": load_aao_index,
    "ENSO": load_enso_index,
}


# ══════════════════════════════════════════════════════════════════════════════
# S3 + Latent Loading (reuses pattern from steer_aurora.py)
# ══════════════════════════════════════════════════════════════════════════════

def init_s3_client():
    """Initialize S3 client if credentials are available."""
    if not HAS_BOTO3:
        return None
    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
    access_key = os.getenv("UVA_S3_ACCESS_KEY")
    secret_key = os.getenv("UVA_S3_SECRET_KEY")
    if access_key and secret_key:
        client = boto3.client(
            "s3",
            endpoint_url="https://ceph-gw.science.uva.nl:8000",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        print("S3 Client initialized.")
        return client
    return None


def load_single_latent(
    date_str: str,
    layer: str = "encoder_2",
    hhmm: str = "0000",
    s3_client=None,
) -> torch.Tensor | None:
    """Load a single latent tensor for one date, checking local paths then S3."""
    filename = f"latent_{date_str}_{hhmm}_{layer}.pt"

    possible_paths = [
        Path(filename),
        Path("thesis/results") / filename,
        Path(os.environ.get("TMPDIR", "/tmp/ekasteleyn")) / "aurora_hres_latents" / filename,
    ]

    file_path = None
    for p in possible_paths:
        if p.exists():
            file_path = p
            break

    if file_path is None and s3_client is not None:
        s3_key = f"aurora_hres_validation/{filename}"
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                s3_client.download_file("ekasteleyn-aurora-predictions", s3_key, tmp.name)
                file_path = Path(tmp.name)
        except Exception:
            pass

    if file_path is None or not file_path.exists():
        return None

    try:
        t = torch.load(file_path, weights_only=True, map_location="cpu").float()
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None
    finally:
        # Clean up temp files
        if str(file_path).startswith("/tmp") and "tmp" in file_path.name:
            file_path.unlink(missing_ok=True)

    return t


# ══════════════════════════════════════════════════════════════════════════════
# Spatial Masking (same logic as steer_aurora.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_polar_lat_mask(lat_size: int, lat_min: float = 60.0, hemisphere: str = "both") -> torch.Tensor:
    """Create a 1D boolean mask over latent latitude rows."""
    latitudes = torch.linspace(90.0, -90.0, steps=lat_size)
    if hemisphere == "north":
        return latitudes >= lat_min
    if hemisphere == "south":
        return latitudes <= -lat_min
    return latitudes.abs() >= lat_min


def apply_spatial_mask_to_vector(
    vec: torch.Tensor,
    mask_region: str = "polar",
    polar_lat_min: float = 60.0,
    hemisphere: str = "both",
) -> torch.Tensor:
    """Apply spatial mask to a steering vector.

    Expected Aurora encoder latent shape: [1, 16200, C] (90 lat × 180 lon)
    """
    if mask_region == "none":
        return vec

    if vec.ndim != 3:
        print(f"Warning: unexpected vec ndim={vec.ndim}; skipping spatial mask.")
        return vec

    _, seq_len, _ = vec.shape
    if seq_len == 16200 and mask_region == "polar":
        lat_size, lon_size = 90, 180
        lat_mask_1d = build_polar_lat_mask(lat_size, lat_min=polar_lat_min, hemisphere=hemisphere)
        spatial_mask = lat_mask_1d.unsqueeze(1).expand(lat_size, lon_size).reshape(1, seq_len, 1)
        spatial_mask = spatial_mask.to(dtype=vec.dtype, device=vec.device)
        return vec * spatial_mask

    print(f"Warning: could not apply mask for seq_len={seq_len}, skipping mask.")
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# Core: Build Dataset + Train Probe
# ══════════════════════════════════════════════════════════════════════════════

def build_probe_dataset(
    dates_csv: str,
    phenomenon: str,
    layer: str = "encoder_2",
    hhmm: str = "0000",
    s3_client=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y) dataset by pairing latent tensors with index values.

    Returns:
        X: array of shape (N, D) — flattened latent features
        y: array of shape (N,) — ground-truth index values
        valid_dates: list of date strings that were successfully loaded
    """
    # Load the CSV (may contain Active/Neutral dates for various phenomena)
    df = pd.read_csv(dates_csv)

    # If CSV has Year/Month/Day columns (target_dates format), build date strings
    if {"Year", "Month", "Day"}.issubset(df.columns):
        if "Phenomenon" in df.columns:
            df = df[df["Phenomenon"] == phenomenon]
        df["date"] = pd.to_datetime(
            df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"})
        )
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    elif "init_date" in df.columns:
        df["date"] = pd.to_datetime(df["init_date"])
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    else:
        raise ValueError(f"CSV {dates_csv} has no recognisable date columns")

    # De-duplicate dates (same day may appear for multiple types)
    df = df.drop_duplicates(subset=["date_str"])

    # Load ground-truth index
    index_df = INDEX_LOADERS[phenomenon]()
    index_df["date"] = pd.to_datetime(index_df["date"])

    # Merge
    merged = df.merge(index_df, on="date", how="inner")
    print(f"  Merged: {len(merged)} dates with both latents & index values")

    X_list, y_list, valid_dates = [], [], []
    for _, row in merged.iterrows():
        latent = load_single_latent(row["date_str"], layer=layer, hhmm=hhmm, s3_client=s3_client)
        if latent is None:
            continue

        # Flatten to 1D feature vector
        flat = latent.reshape(-1).numpy()
        X_list.append(flat)
        y_list.append(row["index_value"])
        valid_dates.append(row["date_str"])

    if not X_list:
        raise RuntimeError("No valid latent–index pairs found. Check file paths and S3.")

    X = np.stack(X_list, axis=0)  # (N, D)
    y = np.array(y_list)           # (N,)
    print(f"  Dataset built: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Index range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}, std={y.std():.3f}")
    return X, y, valid_dates


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float] | None = None,
    n_folds: int = 5,
) -> tuple[RidgeCV, dict]:
    """Train a Ridge regression probe with cross-validated regularisation.

    Returns:
        model: trained RidgeCV
        metrics: dict with R², RMSE, best alpha, etc.
    """
    if alphas is None:
        alphas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]

    # Standardise features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # RidgeCV with built-in LOO or GCV for alpha selection
    probe = RidgeCV(alphas=alphas, store_cv_results=True)
    probe.fit(X_scaled, y)

    y_pred = probe.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Also do a proper k-fold cross-validation for honest generalisation estimate
    kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)
    cv_r2s = []
    for train_idx, val_idx in kf.split(X_scaled):
        from sklearn.linear_model import Ridge
        fold_model = Ridge(alpha=probe.alpha_)
        fold_model.fit(X_scaled[train_idx], y[train_idx])
        fold_pred = fold_model.predict(X_scaled[val_idx])
        cv_r2s.append(r2_score(y[val_idx], fold_pred))

    metrics = {
        "train_r2": float(r2),
        "train_rmse": float(rmse),
        "best_alpha": float(probe.alpha_),
        "cv_r2_mean": float(np.mean(cv_r2s)),
        "cv_r2_std": float(np.std(cv_r2s)),
        "n_samples": len(y),
        "n_features": X.shape[1],
    }

    print(f"\n  ══ Probe Results ══")
    print(f"  Best λ (alpha):   {metrics['best_alpha']:.2e}")
    print(f"  Train R²:         {metrics['train_r2']:.4f}")
    print(f"  Train RMSE:       {metrics['train_rmse']:.4f}")
    print(f"  {n_folds}-Fold CV R²:    {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(f"  Samples / Feats:  {metrics['n_samples']} / {metrics['n_features']}")

    # Attach scaler to probe for later use
    probe._scaler = scaler
    return probe, metrics


def extract_steering_vector(
    probe: RidgeCV,
    latent_shape: tuple,
) -> torch.Tensor:
    """Extract the weight vector W from the trained probe and reshape it
    back to the original latent tensor geometry.

    The scaler inverse-transforms the weights so the vector operates
    in the original (unscaled) latent space.

    Returns:
        Steering vector with shape = latent_shape
    """
    # probe.coef_ has shape (D,) for single-target regression
    w_scaled = probe.coef_  # in scaled feature space

    # Transform back to original space:  w_orig = w_scaled / scale
    scaler = probe._scaler
    w_original = w_scaled / scaler.scale_

    # Reshape to latent geometry
    w_tensor = torch.from_numpy(w_original.astype(np.float32)).reshape(latent_shape)

    # Normalise to unit norm so alpha directly controls magnitude
    w_norm = torch.norm(w_tensor).item()
    if w_norm > 0:
        w_tensor = w_tensor / w_norm

    print(f"  Steering vector shape: {w_tensor.shape}")
    print(f"  Pre-normalisation ||W||: {w_norm:.6f}")
    return w_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Inference Hook (same pattern as steer_aurora.py)
# ══════════════════════════════════════════════════════════════════════════════

def make_intervention_hook(steering_vec: torch.Tensor, alpha: float = 1.0):
    """Create a forward hook that injects α·W into the latent activations."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output

        s_vec = steering_vec.to(dtype=x.dtype, device=x.device)
        new_x = x + (alpha * s_vec)

        if is_tuple:
            return (new_x,) + output[1:]
        return new_x
    return hook


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Linear Probe Steering: train a ridge probe on Aurora latents, "
                    "extract W as a continuous steering vector, and run steered inference."
    )

    # ── Data / Probe Training ────────────────────────────────────────────────
    parser.add_argument("--phenomenon", type=str, default="AO",
                        choices=["AO", "AAO", "ENSO"],
                        help="Climate phenomenon to probe")
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV with dates to use for training the probe "
                             "(e.g. target_dates_ao_81.csv)")
    parser.add_argument("--layer", type=str, default="encoder_2",
                        help="Latent layer to probe (default: encoder_2)")
    parser.add_argument("--hhmm", type=str, default="0000",
                        help="Init-time tag on latent filenames (default: 0000)")
    parser.add_argument("--ridge-alphas", type=float, nargs="+",
                        default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5],
                        help="Regularisation strengths for RidgeCV")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--name-suffix", type=str, default="",
                        help="Suffix for output filenames")

    # ── Spatial Masking ──────────────────────────────────────────────────────
    parser.add_argument("--mask-region", type=str, default="none",
                        choices=["none", "polar"],
                        help="Spatial mask region for steering vector")
    parser.add_argument("--polar-lat-min", type=float, default=60.0,
                        help="Polar mask starts at |lat| >= this value")
    parser.add_argument("--hemisphere", type=str, default="both",
                        choices=["both", "north", "south"],
                        help="Polar mask hemisphere")

    # ── Steered Inference (optional) ─────────────────────────────────────────
    parser.add_argument("--run-inference", action="store_true",
                        help="Run steered Aurora inference after training")
    parser.add_argument("--steering-alphas", type=float, nargs="+",
                        default=[1.0],
                        help="Steering strengths α for inference")
    parser.add_argument("--base-date", type=str, default=None,
                        help="Base date YYYY-MM-DD for inference")
    parser.add_argument("--init-hour", type=int, default=12,
                        choices=[0, 12],
                        help="Initialization hour for inference")
    parser.add_argument("--steps", type=int, default=12,
                        help="Number of rollout steps (12 = 3 days at 6h)")

    args = parser.parse_args()

    suffix_str = args.name_suffix if args.name_suffix.startswith("_") or args.name_suffix == "" else f"_{args.name_suffix}"
    print("=" * 70)
    print("  LINEAR PROBE STEERING PIPELINE")
    print("=" * 70)
    print(f"  Phenomenon: {args.phenomenon}")
    print(f"  Layer:      {args.layer}")
    print(f"  CSV:        {args.csv}")
    print("=" * 70)

    # ── Step 1: Build Dataset ────────────────────────────────────────────────
    print("\n[1/4] Building probe dataset...")
    s3_client = init_s3_client()
    X, y, valid_dates = build_probe_dataset(
        dates_csv=args.csv,
        phenomenon=args.phenomenon,
        layer=args.layer,
        hhmm=args.hhmm,
        s3_client=s3_client,
    )

    # ── Step 2: Train Probe ──────────────────────────────────────────────────
    print("\n[2/4] Training ridge probe...")
    probe, metrics = train_probe(X, y, alphas=args.ridge_alphas, n_folds=args.n_folds)

    # ── Step 3: Extract Steering Vector ──────────────────────────────────────
    print("\n[3/4] Extracting steering vector W...")

    # Recover original latent shape from first sample
    sample_latent = load_single_latent(valid_dates[0], layer=args.layer, hhmm=args.hhmm, s3_client=s3_client)
    latent_shape = sample_latent.shape
    print(f"  Original latent shape: {latent_shape}")

    steering_vec = extract_steering_vector(probe, latent_shape)

    # Apply spatial mask
    masked_vec = apply_spatial_mask_to_vector(
        steering_vec,
        mask_region=args.mask_region,
        polar_lat_min=args.polar_lat_min,
        hemisphere=args.hemisphere,
    )
    nz_ratio = (masked_vec != 0).float().mean().item()
    print(f"  Mask: region={args.mask_region}, hemisphere={args.hemisphere}, "
          f"lat_min={args.polar_lat_min}. Non-zero fraction={nz_ratio:.4f}")

    # ── Save Outputs ─────────────────────────────────────────────────────────
    output_dir = Path("thesis/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_path = output_dir / f"probe_steering_vector_{args.phenomenon.lower()}{suffix_str}.pt"
    norm_path = output_dir / f"probe_steering_norm_{args.phenomenon.lower()}{suffix_str}.pt"
    metrics_path = output_dir / f"probe_metrics_{args.phenomenon.lower()}{suffix_str}.json"

    torch.save(masked_vec.cpu(), vec_path)
    torch.save(torch.norm(masked_vec, dim=-1).squeeze(0).cpu(), norm_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved steering vector → {vec_path.name}")
    print(f"  Saved steering norm   → {norm_path.name}")
    print(f"  Saved probe metrics   → {metrics_path.name}")

    # ── Step 4: Steered Inference (optional) ─────────────────────────────────
    if not args.run_inference:
        print("\n[4/4] Skipping inference (use --run-inference to enable).")
        print("\nDone!")
        return

    print("\n[4/4] Running steered inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data
    shared_scratch = Path("/scratch-shared/ekasteleyn/aurora_data")
    if shared_scratch.parent.exists():
        download_dir = shared_scratch
    else:
        download_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "aurora_data"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Data dir: {download_dir}")
    download_static(download_dir)

    if args.base_date is None:
        print("  Error: --base-date required for inference.")
        sys.exit(1)

    base_day_str = args.base_date
    print(f"  Base date: {base_day_str}")
    download_data(base_day_str, download_dir)
    if args.init_hour == 0:
        prev_day = (pd.to_datetime(base_day_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day, download_dir)

    batch = prepare_batch(base_day_str, download_dir, init_hour=args.init_hour)
    date_tag = base_day_str.replace("-", "")
    init_tag = f"{args.init_hour:02d}00"

    # Load model
    print(f"  Loading Aurora on {device}...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)

    if isinstance(batch, tuple):
        batch = tuple(t.to(device) if hasattr(t, "to") else t for t in batch)
    else:
        batch = batch.to(device)

    # Base (unsteered) run
    base_filename = f"base_probe_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_alpha_0.0.nc"
    if not os.path.exists(base_filename):
        print(f"  Running base inference (α=0) for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                base_pred = pred
        base_pred = base_pred.to("cpu")
        base_ds = batch_to_dataset(base_pred, step=args.steps)
        base_ds.to_netcdf(base_filename)
        print(f"  Saved → {base_filename}")
    else:
        print(f"  Base output already exists, skipping.")

    # Steered runs
    for alpha_val in args.steering_alphas:
        print(f"\n  Steering with α={alpha_val}...")

        hook_handle = model.backbone.encoder_layers[2].register_forward_hook(
            make_intervention_hook(masked_vec, alpha=alpha_val)
        )

        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                pred_batch = pred

        pred_batch = pred_batch.to("cpu")
        hook_handle.remove()

        ds = batch_to_dataset(pred_batch, step=args.steps)

        lat_tag = str(args.polar_lat_min).replace(".", "p")
        mask_tag = "nomask" if args.mask_region == "none" else f"polar_{args.hemisphere}_lat{lat_tag}"
        out_filename = (
            f"steered_probe_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_"
            f"{mask_tag}_alpha_{alpha_val}.nc"
        )
        ds.to_netcdf(out_filename)
        print(f"  Saved → {out_filename}")

    print("\nDone!")


if __name__ == "__main__":
    main()
