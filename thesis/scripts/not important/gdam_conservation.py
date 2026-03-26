"""
Global Dry Air Mass (GDAM) Conservation Check

Implementation of the GDAM conservation metric from:
"Improving AI weather prediction models using global mass and energy conservation schemes"
(arXiv:2501.05648)

This script calculates the Global Dry Air Mass for ERA5 ground truth and Aurora predictions
to assess mass conservation properties of the AI weather model.

Formula (Section 4.1 / Appendix B):
    M_dry = Σ_i A_i × (P_s,i / g - TCWV_i)

Where:
    - P_s,i  = Surface Pressure (Pa) at pixel i
    - TCWV_i = Total Column Water Vapour (kg/m²) at pixel i
    - A_i    = Area of pixel i (m²)
    - g      = Gravity (9.80665 m/s²)

If TCWV is not available, it is computed from specific humidity:
    TCWV ≈ (1/g) × Σ q × Δp

Surface Pressure Derivation (when SP is absent):
    P_s = P_MSL × exp( -z / (R_d × T_mean) )
    T_mean = T_2m + (Γ × z / (2g))

Author: GDAM Conservation Check Implementation
Compatible with: WeatherBench 2 Zarr datasets & NCAR/CREDIT-physics-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import xarray as xr


# ============================================================================
# Physical Constants
# ============================================================================

GRAVITY = 9.80665       # m/s², standard gravity
EARTH_RADIUS = 6.371e6  # m, mean Earth radius
EXAGRAM = 1e18          # kg, conversion factor to Exagrams
R_DRY = 287.05          # J/(kg·K), specific gas constant for dry air
LAPSE_RATE = 0.0065     # K/m, standard tropospheric lapse rate
C_P = 1004.0            # J/(kg·K), specific heat at constant pressure
L_V = 2.501e6           # J/kg, latent heat of vapourisation


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class ConservationResult:
    """Container for global conservation calculation results."""

    dry_mass_kg: float          # Total dry air mass in kg
    dry_mass_exagram: float     # Total dry air mass in Exagrams (10^18 kg)
    water_mass_kg: float        # Global water mass in kg
    total_energy_joules: float  # Global total energy in J
    total_column_dry_mass: xr.DataArray  # Spatial field (kg/m²)
    tcwv: xr.DataArray          # Total Column Water Vapour field (kg/m²)

    def __repr__(self) -> str:
        return (
            f"ConservationResult(\n"
            f"  dry_mass     = {self.dry_mass_exagram:.6f} Eg ({self.dry_mass_kg:.4e} kg)\n"
            f"  water_mass   = {self.water_mass_kg:.4e} kg\n"
            f"  total_energy = {self.total_energy_joules:.4e} J\n"
            f")"
        )


# Backward-compatible alias
GDAMResult = ConservationResult


@dataclass
class GDAMComparisonResult:
    """Container for comparing conservation metrics between ERA5 and model prediction."""

    era5: ConservationResult
    prediction: ConservationResult
    residual_kg: float
    residual_exagram: float
    relative_residual_pct: float        # Dry-mass percentage relative to ERA5
    water_residual_kg: float            # Water mass residual (kg)
    water_relative_pct: float           # Water mass relative residual (%)
    energy_residual_J: float            # Total energy residual (J)
    energy_relative_pct: float          # Total energy relative residual (%)

    def __repr__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  GLOBAL CONSERVATION CHECK\n"
            f"{'='*60}\n"
            f"\n  [Dry Air Mass]\n"
            f"  ERA5:      {self.era5.dry_mass_exagram:12.6f} Eg\n"
            f"  Aurora:    {self.prediction.dry_mass_exagram:12.6f} Eg\n"
            f"  Residual:  {self.residual_exagram:+12.6f} Eg  ({self.relative_residual_pct:+.6f}%)\n"
            f"\n  [Water Mass]\n"
            f"  ERA5:      {self.era5.water_mass_kg:12.4e} kg\n"
            f"  Aurora:    {self.prediction.water_mass_kg:12.4e} kg\n"
            f"  Residual:  {self.water_residual_kg:+12.4e} kg  ({self.water_relative_pct:+.6f}%)\n"
            f"\n  [Total Energy]\n"
            f"  ERA5:      {self.era5.total_energy_joules:12.4e} J\n"
            f"  Aurora:    {self.prediction.total_energy_joules:12.4e} J\n"
            f"  Residual:  {self.energy_residual_J:+12.4e} J  ({self.energy_relative_pct:+.6f}%)\n"
            f"{'='*60}\n"
        )


# ============================================================================
# Grid Cell Area Calculation
# ============================================================================

def get_grid_cell_area(
    ds: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    earth_radius: float = EARTH_RADIUS
) -> xr.DataArray:
    """
    Compute the area of each grid cell based on latitude coordinates.
    
    For a regular lat/lon grid, the area of each cell is:
        A = R² × Δλ × (sin(φ₂) - sin(φ₁))
    
    Where:
        R  = Earth radius
        Δλ = Longitude spacing in radians
        φ₁, φ₂ = Southern and Northern edges of cell in radians
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing lat/lon coordinates
    lat_name : str
        Name of the latitude coordinate
    lon_name : str
        Name of the longitude coordinate
    earth_radius : float
        Earth radius in meters
        
    Returns
    -------
    xr.DataArray
        Area of each grid cell in m², with dimensions (lat, lon)
    """
    # Get coordinates
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    
    # Determine grid spacing
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)
    
    # Compute edges of each cell (half grid spacing above and below)
    lat_edges_south = lat_rad - dlat_rad / 2
    lat_edges_north = lat_rad + dlat_rad / 2
    
    # Clip to valid range [-π/2, π/2]
    lat_edges_south = np.clip(lat_edges_south, -np.pi/2, np.pi/2)
    lat_edges_north = np.clip(lat_edges_north, -np.pi/2, np.pi/2)
    
    # Area = R² × Δlon × |sin(lat_north) - sin(lat_south)|
    area_per_lat = (
        earth_radius**2 
        * dlon_rad 
        * np.abs(np.sin(lat_edges_north) - np.sin(lat_edges_south))
    )
    
    # Broadcast to 2D array (lat, lon)
    # Each longitude has the same area weight at a given latitude
    nlon = len(lon)
    area_2d = np.broadcast_to(area_per_lat[:, np.newaxis], (len(lat), nlon))
    
    # Create DataArray with proper coordinates
    area_da = xr.DataArray(
        area_2d,
        dims=[lat_name, lon_name],
        coords={lat_name: lat, lon_name: lon},
        name="grid_cell_area",
        attrs={"units": "m²", "long_name": "Grid cell area"}
    )
    
    return area_da


def get_cosine_weights(
    ds: xr.Dataset,
    lat_name: str = "latitude"
) -> xr.DataArray:
    """
    Simple cosine-latitude weighting for global averaging.
    
    This is a simpler alternative to full area calculation,
    useful for normalized metrics.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing latitude coordinate
    lat_name : str
        Name of latitude coordinate
        
    Returns
    -------
    xr.DataArray
        Cosine weights by latitude
    """
    lat = ds[lat_name]
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.sum()  # Normalize
    return weights


# ============================================================================
# Surface Pressure Derivation (Hypsometric Equation)
# ============================================================================

def derive_surface_pressure(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    msl_names: tuple[str, ...] = ("mean_sea_level_pressure", "msl"),
    t2m_names: tuple[str, ...] = ("2m_temperature", "t2m"),
    z_names: tuple[str, ...] = ("geopotential_at_surface", "z"),
    gravity: float = GRAVITY,
    r_dry: float = R_DRY,
    lapse_rate: float = LAPSE_RATE,
) -> xr.DataArray:
    """
    Derive surface pressure from Mean Sea Level Pressure using the
    hypsometric equation with a lapse-rate-corrected mean temperature.

    Formula:
        T_mean = T_2m + (Γ × z) / (2g)
        P_s    = P_MSL × exp( -z / (R_d × T_mean) )

    Where z is the surface geopotential (m²/s²), Γ the tropospheric lapse
    rate (K/m), and g the gravitational acceleration.

    Parameters
    ----------
    ds : xr.Dataset
        Dynamic dataset containing MSL pressure and 2 m temperature.
    ds_static : xr.Dataset
        Static/invariant dataset containing surface geopotential.
    msl_names : tuple[str, ...]
        Candidate variable names for Mean Sea Level Pressure.
    t2m_names : tuple[str, ...]
        Candidate variable names for 2 m temperature.
    z_names : tuple[str, ...]
        Candidate variable names for surface geopotential in *ds_static*.
    gravity : float
        Gravitational acceleration (m/s²).
    r_dry : float
        Specific gas constant for dry air (J/(kg·K)).
    lapse_rate : float
        Standard tropospheric lapse rate (K/m).

    Returns
    -------
    xr.DataArray
        Derived surface pressure (Pa) with spatial dimensions.

    Raises
    ------
    ValueError
        If any required variable cannot be found.
    """
    # --- locate MSL pressure ---
    msl = None
    for name in msl_names:
        if name in ds.data_vars:
            msl = ds[name]
            break
    if msl is None:
        raise ValueError(
            f"No MSL pressure found in dynamic dataset. "
            f"Tried {msl_names}. Available: {list(ds.data_vars)}"
        )

    # --- locate 2 m temperature ---
    t2m = None
    for name in t2m_names:
        if name in ds.data_vars:
            t2m = ds[name]
            break
    if t2m is None:
        raise ValueError(
            f"No 2 m temperature found in dynamic dataset. "
            f"Tried {t2m_names}. Available: {list(ds.data_vars)}"
        )

    # --- locate surface geopotential ---
    z_sfc = None
    for name in z_names:
        if name in ds_static.data_vars:
            z_sfc = ds_static[name]
            break
    if z_sfc is None:
        raise ValueError(
            f"No surface geopotential found in static dataset. "
            f"Tried {z_names}. Available: {list(ds_static.data_vars)}"
        )

    # Squeeze out any singleton time dimension in the static field
    if "time" in z_sfc.dims:
        z_sfc = z_sfc.isel(time=0, drop=True)
    if "valid_time" in z_sfc.dims:
        z_sfc = z_sfc.isel(valid_time=0, drop=True)

    # Align grids if necessary (e.g. 721 vs 720 latitudes)
    lat_name = "latitude"
    if lat_name in z_sfc.dims and lat_name in msl.dims:
        if z_sfc.sizes[lat_name] != msl.sizes[lat_name]:
            # Crop the larger to match the smaller
            n_target = msl.sizes[lat_name]
            n_static = z_sfc.sizes[lat_name]
            if n_static > n_target:
                z_sfc = z_sfc.isel({lat_name: slice(0, n_target)})
                z_sfc = z_sfc.assign_coords(
                    {lat_name: msl[lat_name].values}
                )
            else:
                # Static is smaller — unusual, but handle gracefully
                msl = msl.isel({lat_name: slice(0, n_static)})
                t2m = t2m.isel({lat_name: slice(0, n_static)})

    # --- compute ---
    # Lapse-rate-corrected mean column temperature (K)
    t_mean = t2m + (lapse_rate * z_sfc) / (2.0 * gravity)

    # Hypsometric equation: P_s = P_MSL × exp(-z / (R_d × T_mean))
    sp = msl * np.exp(-z_sfc / (r_dry * t_mean))

    sp.name = "surface_pressure"
    sp.attrs = {
        "units": "Pa",
        "long_name": "Surface pressure (derived via hypsometric equation)",
    }
    return sp


# ============================================================================
# Total Column Water Vapour Calculation
# ============================================================================

def compute_tcwv_from_specific_humidity(
    q: xr.DataArray,
    ps: xr.DataArray,
    levels: np.ndarray,
    level_dim: str = "level",
    gravity: float = GRAVITY
) -> xr.DataArray:
    """
    Compute Total Column Water Vapour by integrating specific humidity.
    
    This implementation correctly handles:
    - Spatially varying surface pressure ps(lat, lon)
    - Masking of pressure levels below the surface (p_level > p_surface)
    - Partial bottom layer integration when p_surface falls between levels
    
    Formula:
        TCWV = (1/g) × ∫₀^Pₛ q dp
    
    Uses trapezoidal integration consistent with NCAR CREDIT methodology.
    
    Parameters
    ----------
    q : xr.DataArray
        Specific humidity (kg/kg) with dimensions (level, lat, lon) or similar
    ps : xr.DataArray
        Surface pressure (Pa) with dimensions (lat, lon)
    levels : np.ndarray
        Pressure levels in hPa
    level_dim : str
        Name of the vertical level dimension
    gravity : float
        Gravitational acceleration (m/s²)
        
    Returns
    -------
    xr.DataArray
        Total Column Water Vapour (kg/m²)
    """
    # Convert levels to Pa
    levels_pa = levels.astype(np.float64) * 100.0  # hPa to Pa
    
    # Sort levels from top of atmosphere to surface (increasing pressure)
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]
    q_sorted = q.isel({level_dim: sort_idx})
    
    n_levels = len(levels_sorted)
    
    # Get coordinate names for broadcasting
    lat_dim = [d for d in q.dims if d != level_dim][0]
    lon_dim = [d for d in q.dims if d != level_dim][1]
    
    # Convert ps to numpy for vectorized operations, ensure 2D
    ps_np = ps.values
    if ps_np.ndim == 0:
        ps_np = np.array([[ps_np]])
    elif ps_np.ndim == 3:
        # Handle (time, lat, lon) - squeeze or select first time
        if ps_np.shape[0] == 1:
            ps_np = ps_np[0]
        else:
            raise ValueError(
                f"Surface pressure has multiple time steps {ps_np.shape}. "
                "Please select a single time step before computing TCWV."
            )
    elif ps_np.ndim == 1:
        raise ValueError(
            f"Surface pressure is 1D {ps_np.shape}. Expected 2D (lat, lon)."
        )
    
    # Convert q to numpy array (level, lat, lon)
    q_np = q_sorted.values
    
    # Compute mid-level pressures for trapezoidal integration
    # p_mid[k] is the pressure at the interface between level k and k+1
    p_interfaces = np.zeros(n_levels + 1)
    p_interfaces[0] = 0.0  # Top of atmosphere
    for k in range(n_levels - 1):
        p_interfaces[k + 1] = 0.5 * (levels_sorted[k] + levels_sorted[k + 1])
    p_interfaces[n_levels] = levels_sorted[-1]  # Will be replaced by ps
    
    # Initialize TCWV accumulator
    tcwv_np = np.zeros_like(ps_np)
    
    # Vectorized integration over all grid points
    for k in range(n_levels):
        p_top = p_interfaces[k]
        p_bot_nominal = p_interfaces[k + 1]
        
        # For the bottom-most level, the nominal bottom is the level itself
        # The actual bottom is min(ps, next_interface) but we cap at ps
        if k == n_levels - 1:
            # Last level: integrate from interface above to surface pressure
            p_layer_top = p_interfaces[k]
            # Bottom is the surface pressure, but only where ps > p_top
            p_layer_bot = np.maximum(ps_np, p_layer_top)
        else:
            # Interior levels: use interface pressures
            p_layer_top = p_interfaces[k]
            p_layer_bot = p_interfaces[k + 1]
        
        # Compute layer thickness, respecting surface pressure
        # If ps < p_layer_top, this layer is entirely below surface -> dp = 0
        # If ps >= p_layer_bot, full layer thickness
        # If p_layer_top < ps < p_layer_bot, partial layer
        
        # Effective top: always p_layer_top (but layer masked if ps < p_layer_top)
        # Effective bottom: min(ps, p_layer_bot)
        effective_bot = np.minimum(ps_np, p_layer_bot)
        effective_top = p_layer_top
        
        # dp = max(0, effective_bot - effective_top)
        dp = np.maximum(0.0, effective_bot - effective_top)
        
        # Mask where surface pressure is below layer top (subsurface layer)
        mask = ps_np > p_layer_top
        dp = np.where(mask, dp, 0.0)
        
        # Add contribution: q[k] * dp
        tcwv_np += q_np[k, :, :] * dp
    
    # Divide by gravity to get TCWV in kg/m²
    tcwv_np = tcwv_np / gravity
    
    # Create DataArray with proper coordinates
    tcwv = xr.DataArray(
        tcwv_np,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: q[lat_dim], lon_dim: q[lon_dim]},
        name="tcwv",
        attrs={
            "units": "kg/m²",
            "long_name": "Total Column Water Vapour (integrated from specific humidity)",
            "integration_method": "trapezoidal with surface pressure masking"
        }
    )
    
    return tcwv



def get_tcwv(
    ds: xr.Dataset,
    ps: xr.DataArray,
    tcwv_name: Optional[str] = "tcwv",
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None
) -> xr.DataArray:
    """
    Get or compute Total Column Water Vapour.
    
    If TCWV is directly available in the dataset, use it.
    Otherwise, compute from specific humidity.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing either TCWV or specific humidity
    ps : xr.DataArray
        Surface pressure (Pa)
    tcwv_name : str, optional
        Name of TCWV variable if directly available
    q_name : str
        Name of specific humidity variable
    level_dim : str
        Name of vertical level dimension
    levels : np.ndarray, optional
        Pressure levels in hPa (required if computing from q)
        
    Returns
    -------
    xr.DataArray
        Total Column Water Vapour (kg/m²)
    """
    # Check if TCWV is directly available
    if tcwv_name and tcwv_name in ds.data_vars:
        tcwv = ds[tcwv_name]
        print(f"  Using existing TCWV variable: '{tcwv_name}'")
    elif q_name in ds.data_vars:
        print(f"  Computing TCWV from specific humidity '{q_name}'")
        q = ds[q_name]
        
        # Get levels from dataset if not provided
        if levels is None:
            if level_dim in ds.coords:
                levels = ds[level_dim].values
            else:
                raise ValueError(
                    f"Pressure levels not found. Provide 'levels' argument or "
                    f"ensure '{level_dim}' coordinate exists in dataset."
                )
        
        tcwv = compute_tcwv_from_specific_humidity(
            q=q, ps=ps, levels=levels, level_dim=level_dim
        )
    else:
        raise ValueError(
            f"Cannot compute TCWV: neither '{tcwv_name}' nor '{q_name}' found in dataset."
        )
    
    return tcwv


# ============================================================================
# Column Energy Integration
# ============================================================================

def compute_column_energy(
    T: xr.DataArray,
    q: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    phi: xr.DataArray,
    ps: xr.DataArray,
    levels: np.ndarray,
    level_dim: str = "level",
    gravity: float = GRAVITY,
    c_p: float = C_P,
    l_v: float = L_V,
) -> xr.DataArray:
    """
    Compute column-integrated total energy density.

    Integrates the energy expression:
        E = c_p*T + Φ + L_v*q + 0.5*(u² + v²)
    vertically from the top of the atmosphere down to surface pressure,
    using the same trapezoidal rule and surface-pressure masking as
    :func:`compute_tcwv_from_specific_humidity`.

    Parameters
    ----------
    T : xr.DataArray
        Temperature (K) with dims (level, lat, lon).
    q : xr.DataArray
        Specific humidity (kg/kg), same shape.
    u, v : xr.DataArray
        Wind components (m/s), same shape.
    phi : xr.DataArray
        Geopotential at pressure levels (m²/s²), same shape.
    ps : xr.DataArray
        Surface pressure (Pa), dims (lat, lon).
    levels : np.ndarray
        Pressure levels in hPa.
    level_dim : str
        Name of the vertical dimension.
    gravity : float
        Gravitational acceleration (m/s²).
    c_p : float
        Specific heat at constant pressure (J/(kg·K)).
    l_v : float
        Latent heat of vapourisation (J/kg).

    Returns
    -------
    xr.DataArray
        Column-integrated energy density (J/m²) with dims (lat, lon).
    """
    # Convert levels to Pa and sort top-to-bottom
    levels_pa = levels.astype(np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]

    T_sorted = T.isel({level_dim: sort_idx}).values
    q_sorted = q.isel({level_dim: sort_idx}).values
    u_sorted = u.isel({level_dim: sort_idx}).values
    v_sorted = v.isel({level_dim: sort_idx}).values
    phi_sorted = phi.isel({level_dim: sort_idx}).values

    n_levels = len(levels_sorted)

    # Coordinate names (lat, lon)
    lat_dim = [d for d in T.dims if d != level_dim][0]
    lon_dim = [d for d in T.dims if d != level_dim][1]

    # Surface pressure as 2-D numpy
    ps_np = ps.values
    if ps_np.ndim == 0:
        ps_np = np.array([[ps_np]])
    elif ps_np.ndim == 3:
        if ps_np.shape[0] == 1:
            ps_np = ps_np[0]
        else:
            raise ValueError(
                f"Surface pressure has multiple time steps {ps_np.shape}. "
                "Please select a single time step."
            )

    # Compute interface pressures (same scheme as TCWV)
    p_interfaces = np.zeros(n_levels + 1)
    p_interfaces[0] = 0.0
    for k in range(n_levels - 1):
        p_interfaces[k + 1] = 0.5 * (levels_sorted[k] + levels_sorted[k + 1])
    p_interfaces[n_levels] = levels_sorted[-1]

    # Accumulate column energy
    col_energy = np.zeros_like(ps_np)

    for k in range(n_levels):
        p_top = p_interfaces[k]

        if k == n_levels - 1:
            p_layer_top = p_interfaces[k]
            p_layer_bot = np.maximum(ps_np, p_layer_top)
        else:
            p_layer_top = p_interfaces[k]
            p_layer_bot = p_interfaces[k + 1]

        effective_bot = np.minimum(ps_np, p_layer_bot)
        dp = np.maximum(0.0, effective_bot - p_layer_top)
        mask = ps_np > p_layer_top
        dp = np.where(mask, dp, 0.0)

        # Energy density at this level
        E_k = (
            c_p * T_sorted[k]
            + phi_sorted[k]
            + l_v * q_sorted[k]
            + 0.5 * (u_sorted[k] ** 2 + v_sorted[k] ** 2)
        )

        col_energy += E_k * dp

    # Divide by gravity → J/m²
    col_energy /= gravity

    result = xr.DataArray(
        col_energy,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: T[lat_dim], lon_dim: T[lon_dim]},
        name="column_energy",
        attrs={
            "units": "J/m²",
            "long_name": "Column-integrated total energy density",
            "integration_method": "trapezoidal with surface pressure masking",
        },
    )
    return result


# ============================================================================
# Main Conservation Calculation
# ============================================================================

def calculate_global_conservation(
    ds: xr.Dataset,
    ds_static: Optional[xr.Dataset] = None,
    ps_name: str = "sp",
    tcwv_name: Optional[str] = "tcwv",
    q_name: str = "q",
    t_name: str = "temperature",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    phi_name: str = "geopotential",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    time_idx: Optional[int] = None,
    gravity: float = GRAVITY,
    verbose: bool = True,
) -> ConservationResult:
    """
    Calculate Global Dry Air Mass, Water Mass, and Total Energy for a
    single timestep.

    Formulae
    --------
    Dry Air Mass:   M_dry   = Σ_i A_i × (P_s,i / g − TCWV_i)
    Water Mass:     M_water = Σ_i A_i × TCWV_i
    Total Energy:   TE      = (1/g) × Σ_i A_i × ∫₀^{P_s} E dp
                    with E = c_p T + Φ + L_v q + ½(u² + v²)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing atmospheric variables.
    ds_static : xr.Dataset, optional
        Static dataset with surface geopotential (for SP derivation).
    ps_name, tcwv_name, q_name : str
        Variable names for surface pressure, TCWV, and specific humidity.
    t_name, u_name, v_name, phi_name : str
        Variable names for temperature, wind components, and geopotential
        at pressure levels.
    lat_name, lon_name, level_dim : str
        Coordinate / dimension names.
    levels : np.ndarray, optional
        Pressure levels in hPa.
    time_idx : int, optional
        Time index to select.
    gravity : float
        Gravitational acceleration (m/s²).
    verbose : bool
        Print progress information.

    Returns
    -------
    ConservationResult
    """
    if verbose:
        print("\n  Computing global conservation metrics …")

    # Select single timestep if needed
    if time_idx is not None and "time" in ds.dims:
        ds = ds.isel(time=time_idx)
        if verbose and "time" in ds.coords:
            print(f"  Selected time index {time_idx}: {ds.time.values}")

    # ------------------------------------------------------------------
    # Locate or derive surface pressure
    # ------------------------------------------------------------------
    sp_candidates = ["sp", "surface_pressure", "ps"]

    ps_found = None
    for cand in ([ps_name] if ps_name not in sp_candidates else sp_candidates):
        if cand in ds.data_vars:
            ps_found = cand
            break
    if ps_found is None and ps_name not in sp_candidates:
        for cand in sp_candidates:
            if cand in ds.data_vars:
                ps_found = cand
                break

    if ps_found is not None:
        ps = ds[ps_found]
        if verbose:
            print(f"  Using surface pressure variable: '{ps_found}'")
    else:
        if ds_static is None:
            raise ValueError(
                f"Surface pressure (tried {sp_candidates}) is missing from "
                f"the dataset and no ds_static was provided to derive it. "
                f"Available variables: {list(ds.data_vars)}"
            )
        if verbose:
            print("  Surface pressure missing — deriving via hypsometric equation …")
        ps = derive_surface_pressure(ds, ds_static)

    # Ensure Pa
    ps_mean = float(ps.mean())
    if ps_mean < 2000:
        if verbose:
            print(f"  Converting surface pressure from hPa to Pa (mean: {ps_mean:.1f})")
        ps = ps * 100.0

    # Resolve pressure levels
    if levels is None:
        if level_dim in ds.coords:
            levels = ds[level_dim].values
        else:
            raise ValueError(
                f"Pressure levels not found. Provide 'levels' argument or "
                f"ensure '{level_dim}' coordinate exists in dataset."
            )

    # ------------------------------------------------------------------
    # TCWV  (reused from existing logic)
    # ------------------------------------------------------------------
    tcwv = get_tcwv(
        ds=ds, ps=ps, tcwv_name=tcwv_name, q_name=q_name,
        level_dim=level_dim, levels=levels,
    )

    # ------------------------------------------------------------------
    # Grid cell areas
    # ------------------------------------------------------------------
    area = get_grid_cell_area(ds, lat_name=lat_name, lon_name=lon_name)

    # ------------------------------------------------------------------
    # Dry Air Mass
    # ------------------------------------------------------------------
    column_dry_mass = (ps / gravity) - tcwv
    column_dry_mass.attrs = {"units": "kg/m²", "long_name": "Column Dry Air Mass"}

    dry_mass_kg = float((column_dry_mass * area).sum())
    dry_mass_exagram = dry_mass_kg / EXAGRAM

    if verbose:
        print(f"  Dry Air Mass:   {dry_mass_exagram:.6f} Eg ({dry_mass_kg:.4e} kg)")
        expected = 5.13
        if abs(dry_mass_exagram - expected) / expected > 0.1:
            print(f"  ⚠ Warning: Computed value differs by >{10}% from expected ~{expected} Eg")

    # ------------------------------------------------------------------
    # Water Mass
    # ------------------------------------------------------------------
    water_mass_kg = float((tcwv * area).sum())
    if verbose:
        print(f"  Water Mass:     {water_mass_kg:.4e} kg")

    # ------------------------------------------------------------------
    # Total Energy
    # ------------------------------------------------------------------
    # Locate variables for energy calculation; if any are missing we
    # report energy as NaN rather than failing.
    _t = _v_find(ds, t_name)
    _u = _v_find(ds, u_name)
    _v = _v_find(ds, v_name)
    _phi = _v_find(ds, phi_name)

    if _t is not None and _u is not None and _v is not None and _phi is not None:
        col_energy = compute_column_energy(
            T=ds[_t], q=ds[q_name] if q_name in ds else ds[_v_find(ds, q_name)],
            u=ds[_u], v=ds[_v], phi=ds[_phi],
            ps=ps, levels=levels, level_dim=level_dim,
            gravity=gravity,
        )
        total_energy_J = float((area * col_energy).sum())
        if verbose:
            print(f"  Total Energy:   {total_energy_J:.4e} J")
    else:
        total_energy_J = float("nan")
        missing = [n for n, v in [("T", _t), ("u", _u), ("v", _v), ("Φ", _phi)] if v is None]
        if verbose:
            print(f"  Total Energy:   N/A (missing variables: {missing})")

    return ConservationResult(
        dry_mass_kg=dry_mass_kg,
        dry_mass_exagram=dry_mass_exagram,
        water_mass_kg=water_mass_kg,
        total_energy_joules=total_energy_J,
        total_column_dry_mass=column_dry_mass,
        tcwv=tcwv,
    )


def _v_find(ds: xr.Dataset, name: str) -> Optional[str]:
    """Return *name* if present in *ds*, else None."""
    return name if name in ds.data_vars else None


def calculate_gdam(
    ds: xr.Dataset, ds_static=None, **kwargs,
) -> ConservationResult:
    """Backward-compatible wrapper around :func:`calculate_global_conservation`."""
    return calculate_global_conservation(ds, ds_static=ds_static, **kwargs)


def compare_gdam(
    ds_era5: xr.Dataset,
    ds_pred: xr.Dataset,
    ds_static: Optional[xr.Dataset] = None,
    era5_ps_name: str = "sp",
    pred_ps_name: str = "sp",
    era5_tcwv_name: Optional[str] = "tcwv",
    pred_tcwv_name: Optional[str] = None,
    era5_q_name: str = "q",
    pred_q_name: str = "q",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    time_idx: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> GDAMComparisonResult:
    """
    Compare conservation metrics between ERA5 and a model prediction.

    Computes dry mass, water mass, and total energy for both datasets
    and reports residuals.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  GLOBAL CONSERVATION CHECK")
        print("=" * 60)
        print("\n[ERA5 Ground Truth]")

    era5_result = calculate_global_conservation(
        ds=ds_era5, ds_static=ds_static,
        ps_name=era5_ps_name, tcwv_name=era5_tcwv_name,
        q_name=era5_q_name, lat_name=lat_name, lon_name=lon_name,
        level_dim=level_dim, levels=levels, time_idx=time_idx,
        verbose=verbose, **kwargs,
    )

    if verbose:
        print("\n[Aurora Prediction]")

    pred_result = calculate_global_conservation(
        ds=ds_pred, ds_static=ds_static,
        ps_name=pred_ps_name, tcwv_name=pred_tcwv_name,
        q_name=pred_q_name, lat_name=lat_name, lon_name=lon_name,
        level_dim=level_dim, levels=levels, time_idx=time_idx,
        verbose=verbose, **kwargs,
    )

    # Dry-mass residuals
    residual_kg = pred_result.dry_mass_kg - era5_result.dry_mass_kg
    residual_exagram = pred_result.dry_mass_exagram - era5_result.dry_mass_exagram
    relative_residual_pct = 100.0 * residual_kg / era5_result.dry_mass_kg

    # Water-mass residuals
    water_residual = pred_result.water_mass_kg - era5_result.water_mass_kg
    water_rel = (
        100.0 * water_residual / era5_result.water_mass_kg
        if era5_result.water_mass_kg != 0 else float("nan")
    )

    # Energy residuals
    energy_residual = pred_result.total_energy_joules - era5_result.total_energy_joules
    energy_rel = (
        100.0 * energy_residual / era5_result.total_energy_joules
        if era5_result.total_energy_joules != 0 else float("nan")
    )

    comparison = GDAMComparisonResult(
        era5=era5_result,
        prediction=pred_result,
        residual_kg=residual_kg,
        residual_exagram=residual_exagram,
        relative_residual_pct=relative_residual_pct,
        water_residual_kg=water_residual,
        water_relative_pct=water_rel,
        energy_residual_J=energy_residual,
        energy_relative_pct=energy_rel,
    )

    if verbose:
        print(comparison)

    return comparison


# ============================================================================
# Example / Demo Datasets
# ============================================================================

def create_example_datasets(
    nlat: int = 721,
    nlon: int = 1440,
    nlevels: int = 13
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Create example ERA5 and prediction datasets for testing.
    
    Parameters
    ----------
    nlat : int
        Number of latitude points (721 for 0.25° global)
    nlon : int
        Number of longitude points (1440 for 0.25° global)
    nlevels : int
        Number of pressure levels
        
    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Example ERA5 and prediction datasets
    """
    # Standard 0.25° grid
    lat = np.linspace(90, -90, nlat)
    lon = np.linspace(0, 359.75, nlon)
    
    # Standard pressure levels (hPa) - subset for Aurora
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])[:nlevels]
    
    # Create realistic surface pressure field (Pa)
    # Mean sea level pressure with some spatial variation
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    ps_base = 101325.0  # Standard atmosphere in Pa
    ps_era5 = ps_base + 1000 * np.cos(np.deg2rad(lat_2d)) * np.sin(np.deg2rad(lon_2d))
    
    # Add small perturbation for prediction (mass conservation error)
    ps_pred = ps_era5 * (1 + 0.001 * np.random.randn(*ps_era5.shape))
    
    # Create specific humidity field (kg/kg)
    # Realistic profile decreasing with height
    q_base = 0.01 * np.exp(-np.arange(nlevels) / 3)  # Vertical profile
    q_era5 = np.broadcast_to(q_base[:, np.newaxis, np.newaxis], (nlevels, nlat, nlon)).copy()
    q_era5 *= (1 + 0.5 * np.cos(np.deg2rad(lat_2d)))  # Latitude variation

    q_pred = q_era5 * (1 + 0.01 * np.random.randn(*q_era5.shape))
    q_pred = np.maximum(q_pred, 0)  # Ensure non-negative

    # Create temperature field (K) — realistic ~280 K at surface, decreasing aloft
    t_base = 280.0 - 6.5 * np.arange(nlevels)  # rough lapse rate
    t_base = np.maximum(t_base, 200.0)  # cap at stratospheric temperature
    t_era5 = np.broadcast_to(t_base[:, np.newaxis, np.newaxis], (nlevels, nlat, nlon)).copy()
    t_pred = t_era5 * (1 + 0.001 * np.random.randn(*t_era5.shape))

    # Create wind components (m/s)
    u_era5 = 10.0 * np.cos(np.deg2rad(lat_2d))[np.newaxis, :, :] * np.ones((nlevels, 1, 1))
    v_era5 = 2.0 * np.sin(np.deg2rad(lon_2d))[np.newaxis, :, :] * np.ones((nlevels, 1, 1))
    u_pred = u_era5 * (1 + 0.01 * np.random.randn(*u_era5.shape))
    v_pred = v_era5 * (1 + 0.01 * np.random.randn(*v_era5.shape))

    # Create geopotential at pressure levels (m²/s²)
    # Approximate: φ ≈ g × z, with z ≈ 44331 * (1 - (p/1013.25)^0.19026)
    z_approx = 44_331.0 * (1 - (levels / 1013.25) ** 0.19026)
    phi_era5 = (GRAVITY * z_approx)[:, np.newaxis, np.newaxis] * np.ones((1, nlat, nlon))
    phi_pred = phi_era5.copy()  # geopotential is derived; small perturbation

    # Create ERA5 dataset
    dims3d = ["level", "latitude", "longitude"]
    ds_era5 = xr.Dataset(
        {
            "sp": (["latitude", "longitude"], ps_era5),
            "q": (dims3d, q_era5),
            "temperature": (dims3d, t_era5),
            "u_component_of_wind": (dims3d, u_era5),
            "v_component_of_wind": (dims3d, v_era5),
            "geopotential": (dims3d, phi_era5),
        },
        coords={"latitude": lat, "longitude": lon, "level": levels},
    )

    # Create prediction dataset
    ds_pred = xr.Dataset(
        {
            "sp": (["latitude", "longitude"], ps_pred),
            "q": (dims3d, q_pred),
            "temperature": (dims3d, t_pred),
            "u_component_of_wind": (dims3d, u_pred),
            "v_component_of_wind": (dims3d, v_pred),
            "geopotential": (dims3d, phi_pred),
        },
        coords={"latitude": lat, "longitude": lon, "level": levels},
    )

    return ds_era5, ds_pred


# ============================================================================
# Unit Tests for TCWV Integration
# ============================================================================

def test_tcwv_integration():
    """
    Test that TCWV integration correctly handles surface pressure.
    
    Run with:
        python -c "from gdam_conservation import test_tcwv_integration; test_tcwv_integration()"
    """
    print("\n" + "="*60)
    print("  TCWV INTEGRATION UNIT TESTS")
    print("="*60)
    
    # Standard pressure levels (hPa) - similar to Aurora
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    n_levels = len(levels)
    
    # Create a simple 3x3 grid
    lat = np.array([45.0, 46.0, 47.0])
    lon = np.array([0.0, 1.0, 2.0])
    
    # Create uniform specific humidity profile: q = 0.005 kg/kg at all levels
    q_uniform = 0.005
    q_data = np.full((n_levels, 3, 3), q_uniform)
    q = xr.DataArray(
        q_data,
        dims=["level", "latitude", "longitude"],
        coords={"level": levels, "latitude": lat, "longitude": lon}
    )
    
    # =========================================================================
    # Test 1: Sea level surface pressure (1013.25 hPa)
    # =========================================================================
    print("\n[Test 1] Sea level surface pressure (1013.25 hPa)")
    ps_sea = xr.DataArray(
        np.full((3, 3), 101325.0),  # 1013.25 hPa in Pa
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    
    tcwv_sea = compute_tcwv_from_specific_humidity(q, ps_sea, levels)
    tcwv_mean_sea = float(tcwv_sea.mean())
    
    # Expected: integral from 0 to ~1013 hPa
    # Approximate: q * dp / g ≈ 0.005 * 101325 / 9.80665 ≈ 51.6 kg/m²
    # (This is a rough approximation; actual value depends on layer scheme)
    print(f"  TCWV at sea level: {tcwv_mean_sea:.4f} kg/m²")
    
    # =========================================================================
    # Test 2: Tibetan Plateau (650 hPa surface pressure)
    # =========================================================================
    print("\n[Test 2] Tibetan Plateau (650 hPa surface pressure)")
    ps_tibet = xr.DataArray(
        np.full((3, 3), 65000.0),  # 650 hPa in Pa
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    
    tcwv_tibet = compute_tcwv_from_specific_humidity(q, ps_tibet, levels)
    tcwv_mean_tibet = float(tcwv_tibet.mean())
    
    print(f"  TCWV at Tibet:     {tcwv_mean_tibet:.4f} kg/m²")
    
    # Tibet should have LESS TCWV than sea level (shorter column)
    ratio = tcwv_mean_tibet / tcwv_mean_sea
    print(f"  Ratio Tibet/Sea:   {ratio:.4f}")
    
    if ratio < 0.95:
        print("  ✓ PASS: Tibet has less TCWV than sea level (as expected)")
    else:
        print("  ✗ FAIL: Tibet should have significantly less TCWV!")
    
    # =========================================================================
    # Test 3: Zero humidity everywhere
    # =========================================================================
    print("\n[Test 3] Zero humidity everywhere")
    q_zero = xr.DataArray(
        np.zeros((n_levels, 3, 3)),
        dims=["level", "latitude", "longitude"],
        coords={"level": levels, "latitude": lat, "longitude": lon}
    )
    
    tcwv_zero = compute_tcwv_from_specific_humidity(q_zero, ps_sea, levels)
    tcwv_max_zero = float(np.abs(tcwv_zero).max())
    
    print(f"  TCWV max:          {tcwv_max_zero:.6e} kg/m²")
    
    if tcwv_max_zero < 1e-10:
        print("  ✓ PASS: TCWV is zero when q=0")
    else:
        print("  ✗ FAIL: TCWV should be exactly zero!")
    
    # =========================================================================
    # Test 4: Surface pressure between levels (937.5 hPa, between 925 and 950)
    # =========================================================================
    print("\n[Test 4] Partial bottom layer (937.5 hPa, between 925 and 950 hPa)")
    ps_partial = xr.DataArray(
        np.full((3, 3), 93750.0),  # 937.5 hPa in Pa
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    
    tcwv_partial = compute_tcwv_from_specific_humidity(q, ps_partial, levels)
    tcwv_mean_partial = float(tcwv_partial.mean())
    
    print(f"  TCWV at 937.5 hPa: {tcwv_mean_partial:.4f} kg/m²")
    
    # Should be between sea level and some lower value
    if tcwv_mean_partial < tcwv_mean_sea and tcwv_mean_partial > tcwv_mean_tibet:
        print("  ✓ PASS: Partial layer TCWV is between Tibet and sea level")
    else:
        print("  ✗ FAIL: Partial layer ordering incorrect!")
    
    # =========================================================================
    # Test 5: Spatially varying surface pressure
    # =========================================================================
    print("\n[Test 5] Spatially varying surface pressure")
    ps_varying = xr.DataArray(
        np.array([[65000, 80000, 101325],
                  [70000, 90000, 100000],
                  [75000, 95000, 98000]], dtype=np.float64),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    
    tcwv_varying = compute_tcwv_from_specific_humidity(q, ps_varying, levels)
    
    # Check that lower surface pressure -> lower TCWV
    tcwv_low_ps = float(tcwv_varying.isel(latitude=0, longitude=0))  # 650 hPa
    tcwv_high_ps = float(tcwv_varying.isel(latitude=0, longitude=2))  # 1013 hPa
    
    print(f"  TCWV at ps=650 hPa:  {tcwv_low_ps:.4f} kg/m²")
    print(f"  TCWV at ps=1013 hPa: {tcwv_high_ps:.4f} kg/m²")
    
    if tcwv_low_ps < tcwv_high_ps:
        print("  ✓ PASS: Spatially varying ps produces correct TCWV pattern")
    else:
        print("  ✗ FAIL: Higher ps should have higher TCWV!")
    
    print("\n" + "="*60)
    print("  TESTS COMPLETE")
    print("="*60 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function demonstrating GDAM calculation."""
    
    parser = argparse.ArgumentParser(
        description="Calculate Global Dry Air Mass conservation check"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demonstration with synthetic data"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run TCWV integration unit tests"
    )
    args = parser.parse_args()
    
    if args.test:
        test_tcwv_integration()
        return
    
    # Always demo mode — real data should use the runner script
    print("\n" + "=" * 60)
    print("  GLOBAL CONSERVATION CHECK - DEMONSTRATION MODE")
    print("=" * 60)
    print("\nCreating synthetic test datasets...")

    ds_era5, ds_pred = create_example_datasets()

    print(f"  ERA5 shape: {dict(ds_era5.sizes)}")
    print(f"  Prediction shape: {dict(ds_pred.sizes)}")

    # Run comparison (both synthetic datasets already have SP)
    result = compare_gdam(
        ds_era5=ds_era5,
        ds_pred=ds_pred,
        era5_ps_name="sp",
        pred_ps_name="sp",
        era5_tcwv_name=None,  # Will compute from q
        pred_tcwv_name=None,  # Will compute from q
        era5_q_name="q",
        pred_q_name="q",
        t_name="temperature",
        u_name="u_component_of_wind",
        v_name="v_component_of_wind",
        phi_name="geopotential",
        verbose=True,
    )

    # Additional diagnostic information
    print("\n[Diagnostic Details]")
    print(f"  Grid resolution: {result.era5.tcwv.latitude.size} × {result.era5.tcwv.longitude.size}")
    print(f"  ERA5 TCWV range: [{float(result.era5.tcwv.min()):.2f}, {float(result.era5.tcwv.max()):.2f}] kg/m²")
    print(f"  Pred TCWV range: [{float(result.prediction.tcwv.min()):.2f}, {float(result.prediction.tcwv.max()):.2f}] kg/m²")
    print(f"  ERA5 Water Mass:  {result.era5.water_mass_kg:.4e} kg")
    print(f"  ERA5 Total Energy: {result.era5.total_energy_joules:.4e} J")
    
    return result


if __name__ == "__main__":

    main()
