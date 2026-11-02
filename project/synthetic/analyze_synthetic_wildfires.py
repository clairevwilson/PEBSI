"""
Glacier-wide Kennicott mass balance for each synthetic wildfire run
(PEBSI/run_synthetic_wildfires.py), joined to the design table from
prepare_synthetic_wildfires.py. The effect of each event is its May-Sep 2006
mass balance minus that of run 0, which has no event.
"""
import os
import pandas as pd
import xarray as xr

OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_wildfires/'
DESIGN_FN = os.path.join(OUT_DIR, 'synthetic_wildfire_design.csv')
RESULTS_FN = os.path.join(OUT_DIR, 'synthetic_wildfire_results.csv')
run_fp = os.path.join(OUT_DIR, 'run_{run:03d}_0', 'output.zarr')
SEASON = slice('2006-05-01', '2006-09-30')


def season_mass_balance(run):
    """May-Sep glacier-wide mass balance [m w.e.]: area-weighted sum over points."""
    ds = xr.open_zarr(run_fp.format(run=run), consolidated=False)
    glacier_wide = (ds['mass_balance'] * ds['weight']).sum('point')
    return float(glacier_wide.sel(time=SEASON).sum())


if __name__ == '__main__':
    design = pd.read_csv(DESIGN_FN, index_col='run', parse_dates=['start'])
    design['season_mb'] = [season_mass_balance(run) for run in design.index]
    design['effect'] = design['season_mb'] - design.loc[0, 'season_mb']
    design.to_csv(RESULTS_FN)
    print(f"Saved {RESULTS_FN}")
