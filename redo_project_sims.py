import yaml 
import pickle
import pandas as pd
import simulation as sim

config_fn = 'config_redos.yaml'

site_dict = {
    'wolverine':['N','B','EC'],
    'kahiltna':['K53','K17b'], 
    'kennicott':['GTL','GTH'], # ,'KC31'],
    'lemon_creek':['B','C','D'],
    'taku':['NWB1','TKG3'],
    'gulkana':['AU','B','D']
}

translate_rgi = {
                 'gulkana':{'6': '01.00570', '7':'01.05299'},
                 'kahiltna':{'6':'01.22193','7':'01.04282'},
                 'kennicott':{'6':'01.15645','7':'01.05740'},
                 'wolverine':{'6':'01.09162','7':'01.11350'},
                 'lemon_creek':{'6':'01.01104','7':'01.19406'},
                 'taku':{'6':'01.01390','7':'01.19709'},
                 }

with open('data/best_wind.pkl', 'rb') as f:
    params = pickle.load(f)

sites = [] 
gids = []
wind_factors = []
kps = []
a_ices = []

for glacier in site_dict:
    for site in site_dict[glacier]:
        try:
            wind_factor = float(params[glacier][site])
        except:
            wind_factor = 1

        df = pd.read_csv(f'data/by_glacier/{glacier}/site_constants.csv', index_col='site')
        kp = float(df.loc[site, 'kp'])
        a_ice = float(df.loc[site, 'a_ice'])
        gid = translate_rgi[glacier]['6']

        sites.append(site)
        gids.append(gid)
        kps.append(kp)
        a_ices.append(a_ice)
        wind_factors.append(wind_factor)

configs = {
    'rgi_ids': gids, 
    'sites': sites, 
    'n_points': len(sites),
    'bias_vars': ['temp'],

    'start_date': '2000-04-01',
    'end_date': '2025-09-01',

    'dust_factor': 20,
    'ksp_BC': 1,
    'ksp_OC': 1,
    'kp': kps, 
    'albedo_ice': a_ices,
    'wind_factor': wind_factors, 
    'option_accel_grains': False,
    'option_flat_plates': True,
    'constant_freshgrainsize': 54.5, 

    'debug': True,
    'progress_bar': True,
    'store_data': True,
    'store_vars': ['MB','EB','layers','climate'],

    'output_fp': '../Output/check/',

    # 'deposition_data': 'UKESM',
    # 'ukesm_fp': '../UKESM/dw068_nofires/'

}

with open(config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

if __name__ == '__main__':
    # get command-line args
    args = sim.get_args()
    args.config_fn = config_fn

    # initialize and run the model
    model = sim.PEBSI(args)
    model.run()