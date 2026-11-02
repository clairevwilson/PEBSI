"""
Sample 2010-2025 Gulkana simulation with GLIDE ice dynamics coupled
every year. Records each coupling year's surface mass balance and
dynamics-only mass change per point, plus GLIDE's own thickness and SMB
grids, to dynamics_demo.npz in the run's output folder (next to
output.zarr) for plot_dynamics_demo.py.
"""
import os
import sys
import yaml
import numpy as np
sys.path.append(os.getcwd())
import simulation as sim
import pebsi.connectors.pebsi_glide_connector as connector
from project.parameters import host, HOST_PATHS, GLACIERS, BASE_CONFIG

output_fp = '/ocean/projects/ees260009p/cwilson4/Output/dynamics_demo/'
config_fn = 'dynamics_demo.yaml'

# Gulkana's AD_optimize calibration, run with the physics it was calibrated under
with open(HOST_PATHS[host]['output_fp'] + 'AD_optimize_result.yaml') as f:
    result = yaml.safe_load(f)
g = GLACIERS.index('gulkana')

configs = dict(BASE_CONFIG)
configs.update(HOST_PATHS[host])
configs['kp'] = result['kp'][g]
configs['wind_factor'] = result['wind_factor'][g]
configs['debug'] = True
configs['progress_bar'] = False
configs['store_data'] = True
configs['rgi_ids'] = ['01.00570']
configs['method_distribute'] = 'adaptive'
configs['start_date'] = '2010-04-01 00:00'
configs['end_date'] = '2025-03-31 23:00'
configs['temporal_chunk_years'] = 1
configs['option_dynamics'] = True
configs['dynamics_period_years'] = 1
configs['dynamics_sliding_beta'] = 3.5
# temperate-ice Glen's A (2.4e-24 s-1 Pa-3 = 7.6e-17 yr-1 Pa-3), scaled the same way as defaults.py
configs['dynamics_rheology_B'] = 7.6e-17 ** (-1.0 / 3.0) / (900 * 9.81)
configs['dynamics_dx'] = 100.0
configs['output_freq'] = 'daily'
configs['store_vars'] = ['MB']

# record what each coupling step sees and hands back
record = {}
couple_step = connector.GlideCoupler.couple_step
apply_dynamics_mass_change = connector.apply_dynamics_mass_change
push_forcing = connector.GlideCoupler._push_forcing

def recording_apply(state, mask, dmass, params):
    record['dyn'].append(np.array(dmass) / params.density_water)
    return apply_dynamics_mass_change(state, mask, dmass, params)

def recording_push_forcing(self, region, smb_grid):
    record['smb_grid'].append(np.array(smb_grid))
    return push_forcing(self, region, smb_grid)

def recording_couple_step(self, terrain, point_attrs, state, period_mb_mwe, t_years, dt):
    region = next(iter(self.models))
    record['smb'].append(np.array(period_mb_mwe))
    record['dt'].append(dt)
    record['H_old'].append(self._extract_ice_thickness(region))
    out = couple_step(self, terrain, point_attrs, state, period_mb_mwe, t_years, dt)
    record['H_new'].append(self._extract_ice_thickness(region))
    record['grid'] = self.grids[region]
    return out

connector.apply_dynamics_mass_change = recording_apply
connector.GlideCoupler._push_forcing = recording_push_forcing
connector.GlideCoupler.couple_step = recording_couple_step

for k in ['smb', 'dyn', 'dt', 'H_old', 'H_new', 'smb_grid']:
    record[k] = []
configs['output_fp'] = output_fp
with open(config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

args = sim.get_args()
args.config_fn = config_fn
model = sim.PEBSI(args)
model.initialize()
initial_elev = np.array(model.terrain.elev_n, copy=True)
model.initialize = lambda: None
model.run()
os.remove(config_fn)

grid = record['grid']
np.savez(os.path.join(model.output_fp, 'dynamics_demo.npz'),
         smb=np.array(record['smb']),
         dyn=np.array(record['dyn']),
         dt=np.array(record['dt']),
         H_old=np.array(record['H_old']),
         H_new=np.array(record['H_new']),
         smb_grid=np.array(record['smb_grid']),
         grid_x=grid['x'],
         grid_y=grid['y'],
         grid_crs=grid['crs'].to_wkt(),
         glacier_mask=grid['glacier_mask'],
         lat=np.array(model.terrain.lat_n),
         lon=np.array(model.terrain.lon_n),
         elev_initial=initial_elev,
         elev_final=np.array(model.terrain.elev_n))
print(f'saved {len(record["smb"])} coupling years to {model.output_fp}dynamics_demo.npz')
