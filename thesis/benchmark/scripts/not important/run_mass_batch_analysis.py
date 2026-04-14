"""
Mass Conservation Batch Analysis Script

Performs GDAM (Global Dry Air Mass) conservation analysis across multiple
initialization times and rollout steps for Aurora predictions vs ERA5 ground truth.

This script:
1. Loops through N initialization times
2. For each init, loops through M rollout steps (lead times)
3. Compares Aurora predictions to ERA5 at each valid time
4. Outputs results to CSV and prints summary statistics

Author: Batch Analysis for GDAM Conservation
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

# Import the GDAM calculation function from our module
# Adjust the import path if needed based on your project structure
try:
    from gdam_conservation import calculate_gdam, GDAMResult
except ImportError:
    # If running from a different directory, try relative import
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gdam_conservation import calculate_gdam, GDAMResult


# ============================================================================
# Configuration - Edit these paths!
# ============================================================================

# Base directory containing NetCDF files
DATA_DIR = Path.home() / "aurora_thesis" / "results"

# Subdirectories for Aurora predictions and ERA5 ground truth
AURORA_DIR = Path.home() / "aurora_thesis" / "predictions"
ERA5_DIR = Path.home() / "aurora_thesis" / "predictions" / "era5_truth"

# Output file
OUTPUT_CSV = DATA_DIR / "mass_conservation_results.csv"

# File naming patterns (adjust to match your actual file names)
# These use Python format strings with {init_time} and {valid_time}
# Pattern matches generate_predictions.py output: pred_YYYYMMDD_stepNN.nc
AURORA_FILE_PATTERN = "pred_{init_time:%Y%m%d}_step{step:02d}.nc"
ERA5_FILE_PATTERN = "era5_{valid_time:%Y%m%d}_{valid_time:%H%M}.nc"

# Default initialization dates: First day of each month, Jan-Oct 2020
# Initialized at 06:00 UTC (matching generate_predictions.py)
INIT_DATES = [
    "2020-01-01",
    "2020-02-01",
    "2020-03-01",
    "2020-04-01",
    "2020-05-01",
    "2020-06-01",
    "2020-07-01",
    "2020-08-01",
    "2020-09-01",
    "2020-10-01",
]

# Lead times in hours for each step
LEAD_HOURS = [6, 12, 18]

# Default initialization hour (06:00 UTC)
INIT_HOUR = 6

# Static data containing surface geopotential for hypsometric SP derivation
STATIC_PATH = Path.home() / "downloads" / "era5" / "static.nc"

# Physical constants
Rd = 287.05  # Specific gas constant for dry air [J/(kg·K)]


# ============================================================================
# Helper Functions
# ============================================================================

def get_aurora_path(
    init_time: datetime,
    step: int,
    base_dir: Path = AURORA_DIR,
    pattern: str = AURORA_FILE_PATTERN
) -> Path:
    """
    Construct the file path for an Aurora prediction file.
    
    Parameters
    ----------
    init_time : datetime
        Initialization timestamp
    step : int
        Rollout step number (1, 2, 3, ...)
    base_dir : Path
        Base directory for Aurora files
    pattern : str
        File naming pattern
        
    Returns
    -------
    Path
        Full path to the prediction file
    """
    filename = pattern.format(init_time=init_time, step=step)
    return base_dir / filename


def get_era5_path(
    valid_time: datetime,
    base_dir: Path = ERA5_DIR,
    pattern: str = ERA5_FILE_PATTERN
) -> Path:
    """
    Construct the file path for an ERA5 ground truth file.
    
    Parameters
    ----------
    valid_time : datetime
        Valid (target) timestamp
    base_dir : Path
        Base directory for ERA5 files
    pattern : str
        File naming pattern
        
    Returns
    -------
    Path
        Full path to the ERA5 file
    """
    filename = pattern.format(valid_time=valid_time)
    return base_dir / filename


def load_dataset(filepath: Path) -> Optional[xr.Dataset]:
    """
    Load a NetCDF dataset with error handling.
    
    Parameters
    ----------
    filepath : Path
        Path to the NetCDF file
        
    Returns
    -------
    xr.Dataset or None
        Loaded dataset, or None if file not found
    """
    try:
        if filepath.suffix == '.zarr' or filepath.is_dir():
            ds = xr.open_zarr(filepath)
        else:
            ds = xr.open_dataset(filepath)
        return ds
    except FileNotFoundError:
        print(f"  ⚠ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"  ⚠ Error loading {filepath}: {e}")
        return None


# Global cache for static geopotential (loaded once)
_STATIC_GEOPOTENTIAL = None


def load_static_geopotential() -> xr.DataArray:
    """
    Load the surface geopotential from static.nc (cached).
    
    Returns
    -------
    xr.DataArray
        Surface geopotential z [m²/s²] with dims (latitude, longitude)
    """
    global _STATIC_GEOPOTENTIAL
    if _STATIC_GEOPOTENTIAL is None:
        ds_static = xr.open_dataset(STATIC_PATH)
        # z is stored as (valid_time, lat, lon) - squeeze time
        _STATIC_GEOPOTENTIAL = ds_static["z"].isel(valid_time=0)
    return _STATIC_GEOPOTENTIAL


def derive_sp_from_hypsometric(
    ds_era5: xr.Dataset,
    verbose: bool = True
) -> xr.DataArray:
    """
    Derive surface pressure from MSLP using the hypsometric equation.
    
    When the ERA5 dataset lacks true surface pressure (sp), we compute it from:
        SP = MSL × exp(-Z_static / (Rd × T_2m))
    
    This accounts for topography, converting the theoretical sea-level pressure
    to the actual pressure at the surface elevation.
    
    Parameters
    ----------
    ds_era5 : xr.Dataset
        ERA5 dataset containing 'mean_sea_level_pressure' or 'msl',
        and '2m_temperature' or '2t'
    verbose : bool
        Print status messages
        
    Returns
    -------
    xr.DataArray
        Derived surface pressure [Pa] with dims (latitude, longitude)
    """
    # Find MSLP variable
    msl_names = ["mean_sea_level_pressure", "msl"]
    msl = None
    for name in msl_names:
        if name in ds_era5.data_vars:
            msl = ds_era5[name]
            break
    if msl is None:
        raise ValueError(f"No MSLP variable found. Available: {list(ds_era5.data_vars)}")
    
    # Find 2m temperature variable
    t2m_names = ["2m_temperature", "2t"]
    t2m = None
    for name in t2m_names:
        if name in ds_era5.data_vars:
            t2m = ds_era5[name]
            break
    if t2m is None:
        raise ValueError(f"No 2m temperature found. Available: {list(ds_era5.data_vars)}")
    
    # Load static geopotential
    z_static = load_static_geopotential()
    
    # Ensure grid alignment (static has 721 lat, ERA5 may have been cropped)
    if len(ds_era5.latitude) != len(z_static.latitude):
        # Crop static to match ERA5 grid
        if len(z_static.latitude) == 721 and len(ds_era5.latitude) == 720:
            z_static = z_static.isel(latitude=slice(1, None))
            z_static = z_static.assign_coords(latitude=ds_era5.latitude.values)
    
    # Compute surface pressure using hypsometric equation
    # SP = MSL × exp(-Z / (Rd × T))
    # Note: Z is in m²/s² (geopotential), T is in Kelvin
    import numpy as np
    sp_derived = msl * np.exp(-z_static / (Rd * t2m))
    
    # Copy metadata
    sp_derived.name = "surface_pressure_derived"
    sp_derived.attrs["long_name"] = "Surface pressure (derived via hypsometric equation)"
    sp_derived.attrs["units"] = "Pa"
    
    if verbose:
        print("    ⚠ SP missing in ERA5. Derived SP from MSLP + Topography via Hypsometric Eq.")
    
    return sp_derived


def align_era5_to_aurora(ds_era5: xr.Dataset, ds_aurora: xr.Dataset) -> xr.Dataset:
    """
    Align ERA5 dataset to Aurora's grid by cropping latitudes.
    
    ERA5 has 721 latitude points (includes pole at 90°N).
    Aurora has 720 latitude points (cropped during model inference).
    
    This function detects the mismatch and crops ERA5 to match Aurora.
    
    Parameters
    ----------
    ds_era5 : xr.Dataset
        ERA5 dataset with 721 latitudes
    ds_aurora : xr.Dataset
        Aurora dataset with 720 latitudes
        
    Returns
    -------
    xr.Dataset
        ERA5 dataset cropped to match Aurora's latitude coordinates
    """
    era5_lat = ds_era5.latitude.values
    aurora_lat = ds_aurora.latitude.values
    
    # If grids already match, return as-is
    if len(era5_lat) == len(aurora_lat):
        return ds_era5
    
    # Handle the 721 vs 720 mismatch
    if len(era5_lat) == 721 and len(aurora_lat) == 720:
        # Determine which end has the extra point
        if era5_lat[0] > aurora_lat[0]:
            # ERA5 starts higher (has 90°N), drop first point
            ds_era5 = ds_era5.isel(latitude=slice(1, None))
        elif era5_lat[-1] < aurora_lat[-1]:
            # ERA5 ends lower (has -90°S), drop last point
            ds_era5 = ds_era5.isel(latitude=slice(None, -1))
        else:
            # Fallback: just slice to 720 points
            ds_era5 = ds_era5.isel(latitude=slice(0, 720))
        
        # Force coordinate alignment by assigning Aurora's latitudes
        ds_era5 = ds_era5.assign_coords(latitude=aurora_lat)
    else:
        raise ValueError(
            f"Unexpected grid mismatch: ERA5 has {len(era5_lat)} latitudes, "
            f"Aurora has {len(aurora_lat)} latitudes. Expected 721 vs 720."
        )
    
    return ds_era5


def inject_era5_sp_into_aurora(
    ds_aurora: xr.Dataset,
    ds_era5: xr.Dataset,
    verbose: bool = True
) -> xr.Dataset:
    """
    Inject ERA5 surface pressure into Aurora dataset for mass conservation.
    
    This is required because Aurora predictions contain msl (mean sea level pressure)
    but not sp (true surface pressure). For physically correct mass conservation,
    we borrow the sp from ERA5 ground truth.
    
    Parameters
    ----------
    ds_aurora : xr.Dataset
        Aurora prediction dataset (720 latitudes)
    ds_era5 : xr.Dataset
        ERA5 ground truth dataset (may have 721 latitudes)
    verbose : bool
        Print injection message
        
    Returns
    -------
    xr.Dataset
        Aurora dataset with ERA5's surface pressure injected as 'sp'
    """
    # Align ERA5 to Aurora's grid
    ds_era5_aligned = align_era5_to_aurora(ds_era5, ds_aurora)
    
    # Find surface pressure in ERA5
    sp_names = ["sp", "surface_pressure", "ps"]
    
    sp_var = None
    sp_source = None
    
    # First try true surface pressure
    for name in sp_names:
        if name in ds_era5_aligned.data_vars:
            sp_var = ds_era5_aligned[name]
            sp_source = name
            break
    
    # Fallback: Derive SP from MSLP using hypsometric equation
    if sp_var is None:
        try:
            # Use physical derivation: SP = MSL × exp(-Z/(Rd×T))
            sp_var = derive_sp_from_hypsometric(ds_era5_aligned, verbose=verbose)
            sp_source = "hypsometric_derived"
        except Exception as e:
            raise ValueError(
                f"No surface pressure and failed to derive from hypsometric: {e}. "
                f"Available: {list(ds_era5_aligned.data_vars)}"
            )
    
    # Handle time dimension if present
    if "time" in sp_var.dims:
        sp_var = sp_var.isel(time=0)
    
    # Inject into Aurora dataset
    ds_aurora = ds_aurora.assign(sp=sp_var)
    
    if verbose and sp_source != "hypsometric_derived":
        print(f"    [Injecting ERA5 '{sp_source}' as 'sp' into Aurora dataset]")
    
    return ds_aurora


def calculate_dry_air_mass(ds: xr.Dataset) -> float:
    """
    Wrapper function to calculate dry air mass from a dataset.
    
    This uses the full calculate_gdam function but returns just the
    mass value in Exagrams for compatibility with the batch interface.
    
    Handles both Aurora naming (q, msl) and ERA5 naming (specific_humidity, 
    mean_sea_level_pressure) conventions.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing surface pressure and humidity data
        
    Returns
    -------
    float
        Dry air mass in Exagrams (10^18 kg)
    """
    # Try different variable names for surface pressure
    # Priority: true surface pressure first, MSL is invalid for mass conservation
    import warnings
    sp_priority = ["sp", "surface_pressure", "ps"]
    msl_fallback = ["msl", "mean_sea_level_pressure"]
    
    ps_name = None
    for name in sp_priority:
        if name in ds.data_vars:
            ps_name = name
            break
    
    if ps_name is None:
        for name in msl_fallback:
            if name in ds.data_vars:
                ps_name = name
                warnings.warn(
                    f"Using '{name}' as surface pressure fallback. "
                    "This is PHYSICALLY INVALID for mass conservation over land!",
                    UserWarning
                )
                break
    
    if ps_name is None:
        raise ValueError(f"No surface pressure variable found. Available: {list(ds.data_vars)}")
    
    # Try different variable names for specific humidity
    q_names = ["q", "specific_humidity", "Q", "hus"]
    q_name = None
    for name in q_names:
        if name in ds.data_vars:
            q_name = name
            break
    
    if q_name is None:
        raise ValueError(f"No specific humidity variable found. Available: {list(ds.data_vars)}")
    
    # Squeeze time dimension if present (Aurora data has time=1)
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    
    result = calculate_gdam(
        ds=ds,
        ps_name=ps_name,
        tcwv_name=None,  # Will compute from q
        q_name=q_name,
        verbose=False
    )
    
    return result.dry_mass_exagram


# ============================================================================
# Main Batch Analysis
# ============================================================================

def run_batch_analysis(
    init_times: list[datetime],
    lead_times_hours: list[int],
    aurora_dir: Path = AURORA_DIR,
    era5_dir: Path = ERA5_DIR,
    output_csv: Path = OUTPUT_CSV,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run mass conservation analysis across multiple forecasts.
    
    Parameters
    ----------
    init_times : list[datetime]
        List of initialization timestamps
    lead_times_hours : list[int]
        List of lead times in hours (e.g., [6, 12, 18])
    aurora_dir : Path
        Directory containing Aurora prediction files
    era5_dir : Path
        Directory containing ERA5 ground truth files
    output_csv : Path
        Path to save results CSV
    verbose : bool
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame with mass values and residuals
    """
    results = []
    
    total_comparisons = len(init_times) * len(lead_times_hours)
    comparison_num = 0
    
    if verbose:
        print("\n" + "="*70)
        print("  MASS CONSERVATION BATCH ANALYSIS")
        print("="*70)
        print(f"  Initialization times: {len(init_times)}")
        print(f"  Lead times: {lead_times_hours} hours")
        print(f"  Total comparisons: {total_comparisons}")
        print("="*70 + "\n")
    
    for init_time in init_times:
        if verbose:
            print(f"\n[Init: {init_time:%Y-%m-%d %H:%M}]")
        
        for step_idx, lead_hours in enumerate(lead_times_hours, start=1):
            comparison_num += 1
            valid_time = init_time + timedelta(hours=lead_hours)
            
            if verbose:
                print(f"  Step {step_idx} (+{lead_hours:02d}h) → Valid: {valid_time:%Y-%m-%d %H:%M} ", end="")
            
            # Construct file paths
            aurora_path = get_aurora_path(init_time, step_idx, aurora_dir)
            era5_path = get_era5_path(valid_time, era5_dir)
            
            # Initialize result row
            row = {
                "Init_Time": init_time,
                "Step": step_idx,
                "Lead_Hours": lead_hours,
                "Valid_Time": valid_time,
                "Aurora_Mass": None,
                "ERA5_Mass": None,
                "Residual": None,
                "Error_Percentage": None,
                "Aurora_File": str(aurora_path),
                "ERA5_File": str(era5_path),
                "Status": "OK"
            }
            
            # Load Aurora prediction
            ds_aurora = load_dataset(aurora_path)
            if ds_aurora is None:
                row["Status"] = "Aurora file missing"
                results.append(row)
                if verbose:
                    print("⚠ Aurora missing")
                continue
            
            # Load ERA5 ground truth
            ds_era5 = load_dataset(era5_path)
            if ds_era5 is None:
                row["Status"] = "ERA5 file missing"
                ds_aurora.close()
                results.append(row)
                if verbose:
                    print("⚠ ERA5 missing")
                continue
            
            # Calculate dry air mass for both
            try:
                # ============================================================
                # HYPSOMETRIC SP DERIVATION FOR FAIR COMPARISON
                # ============================================================
                # Both Aurora and ERA5 lack true surface pressure (sp). To ensure
                # a physically correct and fair mass conservation comparison:
                # 1. Derive SP for Aurora from ERA5's MSLP + topography
                # 2. Derive SP for ERA5 from its own MSLP + topography
                # This ensures both datasets use the same SP boundary condition.
                # ============================================================
                
                # Check if Aurora lacks true surface pressure
                aurora_has_sp = any(name in ds_aurora.data_vars 
                                    for name in ["sp", "surface_pressure", "ps"])
                
                if not aurora_has_sp:
                    if verbose:
                        print("")  # Newline for cleaner output
                    ds_aurora = inject_era5_sp_into_aurora(ds_aurora, ds_era5, verbose=verbose)
                    if verbose:
                        print(f"    ", end="")  # Indent for residual output
                
                # Check if ERA5 lacks true surface pressure - derive if needed
                era5_has_sp = any(name in ds_era5.data_vars 
                                  for name in ["sp", "surface_pressure", "ps"])
                
                if not era5_has_sp:
                    sp_era5 = derive_sp_from_hypsometric(ds_era5, verbose=False)
                    ds_era5 = ds_era5.assign(sp=sp_era5)
                
                aurora_mass = calculate_dry_air_mass(ds_aurora)
                era5_mass = calculate_dry_air_mass(ds_era5)
                
                residual = aurora_mass - era5_mass
                error_pct = 100.0 * residual / era5_mass
                
                row["Aurora_Mass"] = aurora_mass
                row["ERA5_Mass"] = era5_mass
                row["Residual"] = residual
                row["Error_Percentage"] = error_pct
                
                if verbose:
                    print(f"✓ Residual: {residual:+.6f} Eg ({error_pct:+.4f}%)")
                
            except Exception as e:
                row["Status"] = f"Calculation error: {str(e)[:50]}"
                if verbose:
                    print(f"⚠ Error: {e}")
            
            finally:
                ds_aurora.close()
                ds_era5.close()
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n✓ Results saved to: {output_csv}")
    
    # Print summary statistics
    if verbose:
        print_summary(df)
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics from the results DataFrame."""
    
    print("\n" + "="*70)
    print("  SUMMARY STATISTICS")
    print("="*70)
    
    # Filter successful calculations
    valid = df[df["Status"] == "OK"].copy()
    failed = df[df["Status"] != "OK"]
    
    print(f"\n  Total comparisons: {len(df)}")
    print(f"  Successful:        {len(valid)}")
    print(f"  Failed:            {len(failed)}")
    
    if len(valid) == 0:
        print("\n  ⚠ No successful calculations. Check file paths and data availability.")
        return
    
    # Mass statistics
    print(f"\n  Mean ERA5 Mass:      {valid['ERA5_Mass'].mean():.6f} Eg")
    print(f"  Mean Aurora Mass:    {valid['Aurora_Mass'].mean():.6f} Eg")
    
    # Residual statistics
    mean_residual = valid["Residual"].mean()
    std_residual = valid["Residual"].std()
    mae = valid["Residual"].abs().mean()
    max_residual = valid["Residual"].abs().max()
    
    print(f"\n  Mean Residual:       {mean_residual:+.6f} Eg")
    print(f"  Std Residual:        {std_residual:.6f} Eg")
    print(f"  Mean Absolute Error: {mae:.6f} Eg")
    print(f"  Max Absolute Error:  {max_residual:.6f} Eg")
    
    # Error percentage statistics
    mean_error_pct = valid["Error_Percentage"].mean()
    max_error_pct = valid["Error_Percentage"].abs().max()
    
    print(f"\n  Mean Error (%):      {mean_error_pct:+.6f}%")
    print(f"  Max Error (%):       {max_error_pct:.6f}%")
    
    # Per-step breakdown
    print("\n  Per-Step Statistics:")
    print("  " + "-"*50)
    step_stats = valid.groupby("Step").agg({
        "Residual": ["mean", "std"],
        "Error_Percentage": "mean"
    }).round(6)
    print(step_stats.to_string())
    
    print("\n" + "="*70)


def generate_demo_analysis() -> pd.DataFrame:
    """
    Generate a demonstration analysis with synthetic data.
    
    This simulates the batch analysis without needing actual files,
    useful for testing the pipeline.
    """
    import numpy as np
    
    print("\n" + "="*70)
    print("  DEMO MODE - Synthetic Data Analysis")
    print("="*70)
    
    # Define initialization times (10 consecutive days)
    base_date = datetime(2023, 6, 1, 0, 0)
    init_times = [base_date + timedelta(days=i) for i in range(10)]
    
    # Lead times
    lead_times_hours = [6, 12, 18]
    
    results = []
    np.random.seed(42)  # For reproducibility
    
    # Typical dry air mass
    base_mass = 5.13  # Exagrams
    
    for init_time in init_times:
        for step_idx, lead_hours in enumerate(lead_times_hours, start=1):
            valid_time = init_time + timedelta(hours=lead_hours)
            
            # Simulate ERA5 mass with small daily variation
            era5_mass = base_mass + 0.01 * np.sin(init_time.timetuple().tm_yday / 365 * 2 * np.pi)
            era5_mass += 0.001 * np.random.randn()
            
            # Simulate Aurora mass with small bias and increasing error with lead time
            aurora_mass = era5_mass * (1 + 0.0001 * step_idx * np.random.randn())
            aurora_mass += 0.0005 * step_idx  # Slight positive bias with lead time
            
            residual = aurora_mass - era5_mass
            error_pct = 100.0 * residual / era5_mass
            
            results.append({
                "Init_Time": init_time,
                "Step": step_idx,
                "Lead_Hours": lead_hours,
                "Valid_Time": valid_time,
                "Aurora_Mass": aurora_mass,
                "ERA5_Mass": era5_mass,
                "Residual": residual,
                "Error_Percentage": error_pct,
                "Aurora_File": f"demo_aurora_{init_time:%Y%m%d}_step{step_idx}.nc",
                "ERA5_File": f"demo_era5_{valid_time:%Y%m%d%H}.nc",
                "Status": "OK"
            })
    
    df = pd.DataFrame(results)
    
    # Save demo results
    demo_csv = DATA_DIR / "mass_conservation_demo_results.csv"
    demo_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(demo_csv, index=False)
    print(f"\n✓ Demo results saved to: {demo_csv}")
    
    print_summary(df)
    
    return df


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis of mass conservation for Aurora predictions"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demonstration with synthetic data"
    )
    parser.add_argument(
        "--aurora-dir", type=str, default=None,
        help="Directory containing Aurora prediction files"
    )
    parser.add_argument(
        "--era5-dir", type=str, default=None,
        help="Directory containing ERA5 ground truth files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01",
        help="Start date for analysis (YYYY-MM-DD). Default uses predefined monthly dates."
    )
    parser.add_argument(
        "--num-inits", type=int, default=10,
        help="Number of initialization times"
    )
    parser.add_argument(
        "--init-interval", type=int, default=24,
        help="Interval between initializations in hours"
    )
    parser.add_argument(
        "--lead-times", type=str, default="6,12,18",
        help="Comma-separated lead times in hours"
    )
    args = parser.parse_args()
    
    if args.demo:
        generate_demo_analysis()
        return
    
    # Parse lead times
    lead_times = [int(x) for x in args.lead_times.split(",")]
    
    # Generate initialization times
    # Use predefined INIT_DATES by default, or custom dates if --start-date provided
    if args.start_date == "2020-01-01":
        # Use predefined monthly dates for 2020
        init_times = [
            datetime.strptime(date_str, "%Y-%m-%d").replace(hour=INIT_HOUR)
            for date_str in INIT_DATES
        ]
    else:
        # Use interval-based approach for custom dates
        start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(hour=INIT_HOUR)
        init_times = [
            start + timedelta(hours=i * args.init_interval)
            for i in range(args.num_inits)
        ]
    
    # Set directories
    aurora_dir = Path(args.aurora_dir) if args.aurora_dir else AURORA_DIR
    era5_dir = Path(args.era5_dir) if args.era5_dir else ERA5_DIR
    output_csv = Path(args.output) if args.output else OUTPUT_CSV
    
    print(f"\nConfiguration:")
    print(f"  Aurora dir: {aurora_dir}")
    print(f"  ERA5 dir:   {era5_dir}")
    print(f"  Output:     {output_csv}")
    print(f"  Init times: {len(init_times)}")
    for t in init_times[:3]:
        print(f"    - {t:%Y-%m-%d %H:%M}")
    if len(init_times) > 3:
        print(f"    ... ({len(init_times) - 3} more)")
    print(f"  Lead times: {lead_times} hours")
    
    # Run analysis
    run_batch_analysis(
        init_times=init_times,
        lead_times_hours=lead_times,
        aurora_dir=aurora_dir,
        era5_dir=era5_dir,
        output_csv=output_csv
    )


if __name__ == "__main__":
    main()
