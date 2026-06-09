import xarray as xr
import pandas as pd
import numpy as np

def get_anomaly(t_var, c_var):
    if 'time' in t_var.coords:
        anom_list = []
        t_var_expanded = t_var if 'time' in t_var.dims else t_var.expand_dims('time')
        for t in t_var_expanded.time:
            t_val = t.values
            t_dt = pd.to_datetime(t_val)
            c_slice = c_var
            if 'dayofyear' in c_slice.coords:
                c_slice = c_slice.sel(dayofyear=t_dt.dayofyear)
            elif 'month' in c_slice.coords:
                c_slice = c_slice.sel(month=t_dt.month)
            if 'hour' in c_slice.coords:
                c_slice = c_slice.sel(hour=t_dt.hour)
            anom_list.append(t_var_expanded.sel(time=t) - c_slice)
        anom = xr.concat(anom_list, dim='time')
        if 'time' not in t_var.dims:
            anom = anom.squeeze('time')
        return anom
    else:
        return t_var - c_var

print("Script compiled successfully!")
