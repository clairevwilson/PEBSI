"""
Glacier-wide Kennicott mass balance for each fixed-timing synthetic heatwave and wildfire run
(PEBSI/run_fixed_heatwaves.py, run_fixed_wildfires.py), joined to the design tables from
prepare_fixed_events.py. The effect of each event is its May-Sep 2006 mass balance minus that of
run 0, which has no event.
"""
import os
import pandas as pd
import xarray as xr

OUT = '/ocean/projects/ees260009p/cwilson4/Output/'
EXPERIMENTS = {'heatwave': OUT + 'fixed_heatwaves/', 'wildfire': OUT + 'fixed_wildfires/'}
SEASON = slice('2006-05-01', '2006-09-30')


def season_mass_balance(out_dir, run):
    """May-Sep glacier-wide mass balance [m w.e.]: area-weighted sum over points."""
    ds = xr.open_zarr(os.path.join(out_dir, f'run_{run:03d}_0', 'output.zarr'), consolidated=False)
    glacier_wide = (ds['mass_balance'] * ds['weight']).sum('point')
    return float(glacier_wide.sel(time=SEASON).sum())


if __name__ == '__main__':
    for name, out_dir in EXPERIMENTS.items():
        design = pd.read_csv(os.path.join(out_dir, f'fixed_{name}_design.csv'), index_col='run',
                             parse_dates=['start'])
        design['season_mb'] = [season_mass_balance(out_dir, run) for run in design.index]
        design['effect'] = design['season_mb'] - design.loc[0, 'season_mb']
        results_fn = os.path.join(out_dir, f'fixed_{name}_results.csv')
        design.to_csv(results_fn)
        print(f'Saved {results_fn}')
