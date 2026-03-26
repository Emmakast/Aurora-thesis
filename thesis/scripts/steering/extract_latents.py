from __future__ import annotations

import gc
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from aurora import Aurora, Batch, Metadata


# ══════════════════════════════════════════════════════════════════════════════
# Custom SDPA for Attention Extraction
# ══════════════════════════════════════════════════════════════════════════════
original_sdpa = F.scaled_dot_product_attention
attention_activations = {}
current_capture_key = None

def custom_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    global current_capture_key, attention_activations
    res = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    
    if current_capture_key is not None:
        scale_factor = scale if scale is not None else (1.0 / (q.size(-1) ** 0.5))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(~attn_mask, float('-inf'))
            else:
                attn = attn + attn_mask
        attn_weights = torch.nn.functional.softmax(attn, dim=-1)
        if current_capture_key not in attention_activations:
            attention_activations[current_capture_key] = []
        # Keep ONLY the last attention weights (overwrites any previous ones for this key)
        attention_activations[current_capture_key] = [attn_weights.detach().cpu()]
        
    return res

if not hasattr(F, '_patched'):
    F.scaled_dot_product_attention = custom_sdpa
    F._patched = True


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr"
    "/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)
STATIC_PATH = os.path.expanduser("~/downloads/era5/static.nc")

SCRIPT_DIR = Path(__file__).resolve().parent
DATES_CSV  = SCRIPT_DIR / "target_dates.csv"
OUTPUT_DIR = Path(f"/scratch-shared/{os.environ.get('USER', 'ekasteleyn')}/aurora_latents")

TARGET_LAYERS: list[tuple[str, int]] = [
    ("perceiver", 0),  
    ("encoder", 0),
    ("encoder", 1),
    ("encoder", 2),
]

VAR_MAP = {
    "2t":  "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t":   "temperature",
    "u":   "u_component_of_wind",
    "v":   "v_component_of_wind",
    "q":   "specific_humidity",
    "z":   "geopotential",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_static(ds: xr.Dataset, static_path: str) -> dict[str, torch.Tensor]:
    ref = ds.isel(latitude=slice(0, 720), longitude=slice(0, 1440))
    static = xr.open_dataset(static_path, engine="netcdf4")
    if "valid_time" in static.dims:
        static = static.isel(valid_time=0)
    static = static.interp(latitude=ref.latitude, longitude=ref.longitude)
    static = static.transpose("latitude", "longitude")
    return {
        "z":   torch.from_numpy(static["z"].values).float(),
        "slt": torch.from_numpy(static["slt"].values).float(),
        "lsm": torch.from_numpy(static["lsm"].values).float(),
    }


def load_batch(
    ds: xr.Dataset,
    static_vars: dict[str, torch.Tensor],
    date_str: str,
) -> Batch | None:
    target_time = pd.to_datetime(f"{date_str}T06:00:00")
    request_times = [target_time - timedelta(hours=6), target_time]

    try:
        frame = ds.sel(time=request_times, method="nearest").load()
        frame = frame.sortby("time")
        frame = frame.isel(latitude=slice(0, 720), longitude=slice(0, 1440))
        if frame.time.size < 2:
            return None
    except Exception:
        return None

    surf = {
        k: torch.from_numpy(
            frame[VAR_MAP[k]]
            .transpose("time", "latitude", "longitude")
            .values
        ).unsqueeze(0).float()
        for k in ["2t", "10u", "10v", "msl"]
    }
    atmos = {
        k: torch.from_numpy(
            frame[VAR_MAP[k]]
            .transpose("time", "level", "latitude", "longitude")
            .values
        ).unsqueeze(0).float()
        for k in ["t", "u", "v", "q", "z"]
    }

    return Batch(
        surf_vars=surf,
        static_vars=static_vars,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=torch.from_numpy(frame.latitude.values),
            lon=torch.from_numpy(frame.longitude.values),
            time=tuple(pd.to_datetime(frame.time.values).to_pydatetime()),
            atmos_levels=tuple(int(l) for l in frame.level.values),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Hook registration
# ══════════════════════════════════════════════════════════════════════════════

def make_attn_pre_hook(key: str):
    def pre_hook(module, args):
        global current_capture_key
        current_capture_key = key
    return pre_hook

def make_attn_post_hook(key: str):
    def post_hook(module, args, output):
        global current_capture_key
        current_capture_key = None
    return post_hook

def register_hooks(
    model: torch.nn.Module,
    target_layers: list[tuple[str, int]],
) -> tuple[dict[str, torch.Tensor], list]:
    activations: dict[str, torch.Tensor] = {}
    handles: list = []

    for part, idx in target_layers:
        key = f"{part}_{idx}"

        def _make_hook(k: str):
            def _hook(_module, _input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                activations[k] = tensor.detach().cpu()
            return _hook

        if part == "perceiver":
            module = model.encoder
            handles.append(module.register_forward_hook(_make_hook(key)))
            
            # For perceiver, we only want the cross-attention between atmospheric levels and latent levels.
            # This happens in the PerceiverAttention module of the last layer of the level_agg.
            last_perceiver_attn = module.level_agg.layers[-1][0]
            attn_key = "perceiver_cross_attn"
            handles.append(last_perceiver_attn.register_forward_pre_hook(make_attn_pre_hook(attn_key)))
            handles.append(last_perceiver_attn.register_forward_hook(make_attn_post_hook(attn_key)))
            
        elif part == "encoder":
            module = model.backbone.encoder_layers[idx]
            # Output of module is after downsampling (or just the final output of the layer)
            handles.append(module.register_forward_hook(_make_hook(key)))
            
            # The last latent layer's attention weights
            last_block_attn = module.blocks[-1].attn
            attn_key = f"encoder_{idx}_attn"
            handles.append(last_block_attn.register_forward_pre_hook(make_attn_pre_hook(attn_key)))
            handles.append(last_block_attn.register_forward_hook(make_attn_post_hook(attn_key)))
        else:
            raise ValueError(f"Unknown part '{part}'")

    return activations, handles


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(DATES_CSV).is_file():
        raise FileNotFoundError(f"Dates CSV not found: {DATES_CSV}")
    if not Path(STATIC_PATH).is_file():
        raise FileNotFoundError(f"Static variables file not found: {STATIC_PATH}")

    dates = pd.read_csv(DATES_CSV)["date"].tolist()
    print(f"Loaded {len(dates)} target dates from {DATES_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Opening Zarr store (lazy) ...")
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    print("Loading and interpolating static variables ...")
    static_vars = load_static(ds, STATIC_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Initialising Aurora (finetuned, 1.3 B params) ...")
    model = Aurora()
    model.load_checkpoint()
    model.eval().to(device)
    print("Model ready.")

    activations, handles = register_hooks(model, TARGET_LAYERS)
    
    print(f"Hooks registered.")

    for n, date_str in enumerate(dates, 1):
        print(f"\n[{n}/{len(dates)}] {date_str}", flush=True)

        batch = load_batch(ds, static_vars, date_str)
        if batch is None:
            print(f"  SKIP - could not load data for {date_str}")
            continue

        batch = batch.to(device)

        with torch.inference_mode():
            model(batch)

        # Save hooked latent features
        for key, tensor in activations.items():
            out_path = OUTPUT_DIR / f"latent_{date_str}_{key}.pt"
            tensor_fp16 = tensor.half()
            torch.save(tensor_fp16, out_path)
            print(f"  {key} latent shape={tuple(tensor_fp16.shape)} dtype={tensor_fp16.dtype} -> {out_path}")

        # Save hooked attention weights
        for key, attn_list in attention_activations.items():
            out_path = OUTPUT_DIR / f"{key}_{date_str}.pt"
            stacked_attn = torch.cat(attn_list, dim=0) if len(attn_list) > 1 else attn_list[0]
            stacked_attn_fp16 = stacked_attn.half()
            torch.save(stacked_attn_fp16, out_path)
            print(f"  {key} attn shape={tuple(stacked_attn_fp16.shape)} dtype={stacked_attn_fp16.dtype} -> {out_path}")

        activations.clear()
        attention_activations.clear()
        del batch
        torch.cuda.empty_cache()
        gc.collect()

    for h in handles:
        h.remove()

    print(f"\nDone. Latents saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
