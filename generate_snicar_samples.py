"""
SNICAR Emulator - Sample Generation
=====================================
Generates Latin Hypercube samples and runs SNICAR for each,
storing inputs and broadband albedo outputs for emulator training.

Outputs:
    snicar_train.npz   - training set inputs + targets
    snicar_val.npz     - validation set
    snicar_test.npz    - test set

Usage:
    python generate_snicar_samples.py
    python generate_snicar_samples.py --n_train 50000 --n_layers 4 --seed 42 --n_jobs 8
"""

import sys, os
sys.path.append('/Users/cvw/local/PEBSI/biosnicar-py/')

import copy
import argparse
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from scipy.stats import qmc
from tqdm import tqdm

from biosnicar.get_albedo import get

# ============================================================
# CONFIGURATION
# ============================================================

YAML_PATH = '/Users/cvw/local/PEBSI/biosnicar-py/biosnicar/inputs.yaml'
SFC_PATH  = 'Data/OP_data/480band/r_sfc/gulkana_cleanice_avg_bba3732.csv'

# Dust bin ratios (total dust -> SNICAR bins)
BIN1 = 0.0751       # 0.05–0.5 µm
BIN2 = 0.20535      # 0.5–1.25 µm
BIN3 = 0.481675     # 1.25–2.5 µm
BIN4 = 0.203775     # 2.5–5 µm
BIN5 = 0.034        # 5–50 µm

# Ice constants (same for all layers)
ICE_CONSTANTS = ['LAYER_TYPE', 'HEX_SIDE', 'HEX_LENGTH', 'SHP_FCTR', 'WATER_COATING', 'CDOM']

# Physical parameter ranges (min, max)
PARAM_RANGES = {
    'grain_size_um':    (55.0,    1500.0),
    'density_kgm3':     (50.0,    700.0),
    'height_m':         (0.01,    0.30),
    'bc_ppb':           (0.0,     5000.0),
    'oc_ppm':           (0.0,     5000.0),
    'dust_ppm':         (0.0,     10000.0),
    'solar_zenith_deg': (0.0,     80.0),
}

LAYER_PARAMS  = ['grain_size_um', 'density_kgm3', 'height_m', 'bc_ppb', 'oc_ppm', 'dust_ppm']
COLUMN_PARAMS = ['solar_zenith_deg']

# ============================================================
# SAMPLING
# ============================================================

def build_column_names(n_layers):
    cols = []
    for i in range(n_layers):
        for param in LAYER_PARAMS:
            cols.append(f'layer{i}_{param}')
    return cols + COLUMN_PARAMS


def lhs_samples(n_samples, n_layers, seed):
    cols = build_column_names(n_layers)
    lo, hi = [], []
    for col in cols:
        key = col.split('_', 1)[1] if col[:5] == 'layer' else col
        lo.append(PARAM_RANGES[key][0])
        hi.append(PARAM_RANGES[key][1])

    sampler = qmc.LatinHypercube(d=len(cols), seed=seed)
    scaled  = qmc.scale(sampler.random(n=n_samples), lo, hi)
    return pd.DataFrame(scaled, columns=cols)


def oversample_extremes(n_samples, n_layers, seed):
    rng     = np.random.default_rng(seed + 999)
    n_edge  = n_samples // 5
    cols    = build_column_names(n_layers)
    records = []

    for _ in range(n_edge):
        row      = {}
        scenario = rng.integers(0, 5)

        for i in range(n_layers):
            row[f'layer{i}_density_kgm3'] = float(min(rng.uniform(50, 700) * (1 + i * 0.1), 700))
            row[f'layer{i}_height_m']     = float(rng.uniform(0.005, 0.10))

            if scenario == 0:   # very clean
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(50, 200))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 10))
                row[f'layer{i}_oc_ppm']        = float(rng.uniform(0, 10))
                row[f'layer{i}_dust_ppm']      = float(rng.uniform(0, 50))
            elif scenario == 1: # very dirty
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(500, 2000))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(2000, 5000))
                row[f'layer{i}_oc_ppm']        = float(rng.uniform(1000, 5000))
                row[f'layer{i}_dust_ppm']      = float(rng.uniform(5000, 10000))
            elif scenario == 2: # large grains
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(1000, 2000))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 500))
                row[f'layer{i}_oc_ppm']        = float(rng.uniform(0, 500))
                row[f'layer{i}_dust_ppm']      = float(rng.uniform(0, 1000))
            elif scenario == 3: # thin top layer
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(500, 1500))
                row[f'layer{i}_height_m']      = float(rng.uniform(0.005, 0.02) if i == 0 else rng.uniform(0.01, 0.10))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 1000))
                row[f'layer{i}_oc_ppm']        = float(rng.uniform(0, 1000))
                row[f'layer{i}_dust_ppm']      = float(rng.uniform(0, 2000))
            else:               # high zenith
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(50, 500))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 200))
                row[f'layer{i}_oc_ppm']        = float(rng.uniform(0, 200))
                row[f'layer{i}_dust_ppm']      = float(rng.uniform(0, 500))

        row['solar_zenith_deg'] = float(rng.uniform(70, 85) if scenario == 4 else rng.uniform(0, 85))
        records.append(row)

    return pd.DataFrame(records, columns=cols)


def enforce_physical_constraints(df, n_layers):
    for i in range(1, n_layers):
        # density and grain size increase with depth
        df[f'layer{i}_density_kgm3']  = np.maximum(df[f'layer{i}_density_kgm3'],  df[f'layer{i-1}_density_kgm3']  * 0.85)
        df[f'layer{i}_grain_size_um'] = np.maximum(df[f'layer{i}_grain_size_um'], df[f'layer{i-1}_grain_size_um'] * 0.90)

    # total column height <= 0.30 m
    height_cols = [f'layer{i}_height_m' for i in range(n_layers)]
    scale = np.minimum(1.0, 0.30 / df[height_cols].sum(axis=1))
    for col in height_cols:
        df[col] *= scale

    return df


# ============================================================
# SNICAR RUNNER
# ============================================================

def load_base_inputs():
    with open(YAML_PATH, 'r') as f:
        return yaml.safe_load(f)


def run_snicar(row, n_layers, base_inputs):
    """
    Configures and runs SNICAR for a single sample row.
    Returns broadband albedo, or NaN on failure.
    """
    inputs = {k: (v.copy() if isinstance(v, dict) else v)
              for k, v in base_inputs.items()}
    # deep copy nested dicts
    inputs = copy.deepcopy(base_inputs) # yaml.safe_load(yaml.dump(base_inputs))

    lheight    = [float(row[f'layer{i}_height_m'])     for i in range(n_layers)]
    ldensity   = [int(row[f'layer{i}_density_kgm3'])   for i in range(n_layers)]
    lgrainsize = [int(row[f'layer{i}_grain_size_um'])  for i in range(n_layers)]
    lBC        = [float(row[f'layer{i}_bc_ppb'])       for i in range(n_layers)]
    lOC        = [float(row[f'layer{i}_oc_ppm'])       for i in range(n_layers)]
    ldust      = np.array([float(row[f'layer{i}_dust_ppm']) for i in range(n_layers)])

    # snap lgrainsize to the lookup grid (files every 1 um up to 1000, then every 500 um)
    lgrainsize = [int(row[f'layer{i}_grain_size_um']) for i in range(n_layers)]
    lgrainsize = [
        int(np.round(r / 500) * 500) if r > 1500 else r
        for r in lgrainsize
    ]

    inputs['ICE']['DZ']  = lheight
    inputs['ICE']['RHO'] = ldensity
    inputs['ICE']['RDS'] = lgrainsize
    inputs['ICE']['LWC'] = [0] * n_layers

    inputs['IMPURITIES']['BC']['CONC']    = lBC
    inputs['IMPURITIES']['OC']['CONC']    = lOC
    inputs['IMPURITIES']['DUST1']['CONC'] = (ldust * BIN1).tolist()
    inputs['IMPURITIES']['DUST2']['CONC'] = (ldust * BIN2).tolist()
    inputs['IMPURITIES']['DUST3']['CONC'] = (ldust * BIN3).tolist()
    inputs['IMPURITIES']['DUST4']['CONC'] = (ldust * BIN4).tolist()
    inputs['IMPURITIES']['DUST5']['CONC'] = (ldust * BIN5).tolist()

    inputs['ICE']['SHP'] = [0] * n_layers
    inputs['ICE']['AR']  = [0] * n_layers
    inputs['ICE']['LAYER_TYPE'][0] = 0
    for var in ICE_CONSTANTS:
        inputs['ICE'][var] = [inputs['ICE'][var][0]] * n_layers

    inputs['PATHS']['SFC']   = SFC_PATH
    inputs['RTM']['SOLZEN']  = float(row['solar_zenith_deg'])
    inputs['RTM']['DIRECT']  = 0

    try:
        albedo, spectral_weights = get(inputs)
        bba = float(np.sum(albedo * spectral_weights) / np.sum(spectral_weights))
        # sanity check
        if not (0.0 <= bba <= 1.0):
            return np.nan
        return bba
    except Exception as e:
        print(f"FAILED: {e} | zen={row['solar_zenith_deg']:.1f} "
          f"rds={[int(row[f'layer{i}_grain_size_um']) for i in range(n_layers)]} "
          f"rho={[int(row[f'layer{i}_density_kgm3']) for i in range(n_layers)]}")
        return np.nan


def run_batch(df, n_layers, base_inputs, desc='Running SNICAR'):
    """Runs SNICAR for every row in df, returns array of broadband albedos."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        results.append(run_snicar(row, n_layers, base_inputs))
    return np.array(results, dtype=np.float32)


def run_batch_parallel(df, n_layers, base_inputs, n_jobs, desc='Running SNICAR'):
    """Parallel version using joblib."""
    rows = [row for _, row in df.iterrows()]
    results = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
        joblib.delayed(run_snicar)(row, n_layers, base_inputs)
        for row in tqdm(rows, desc=desc)
    )
    return np.array(results, dtype=np.float32)


# ============================================================
# SAVE / LOAD
# ============================================================

def save_split(df, bba, path):
    """
    Saves inputs (X) and targets (y) together as a .npz file.
    X shape: (N, n_features), y shape: (N,)
    Also saves column names for reference.
    """
    # drop failed runs
    valid = ~np.isnan(bba)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f'  Warning: {n_dropped} failed SNICAR runs dropped from {path.name}')

    X = df.values[valid].astype(np.float32)
    y = bba[valid]

    np.savez(
        path,
        X=X,
        y=y,
        columns=np.array(df.columns.tolist())
    )
    print(f'  Saved {path}  ({valid.sum():,} samples, {X.shape[1]} features)')


# ============================================================
# MAIN
# ============================================================

def main(n_train, n_val, n_test, n_layers, seed, output_dir, n_jobs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== SNICAR Emulator Sample Generation ===')
    print(f'Layers: {n_layers}  |  Features: {n_layers * len(LAYER_PARAMS) + len(COLUMN_PARAMS)}')
    print(f'Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}')
    print(f'Jobs: {n_jobs}\n')

    base_inputs = load_base_inputs()
    total = n_train + n_val + n_test

    # generate samples
    print('Sampling...')
    df_main = lhs_samples(total, n_layers, seed)
    df_edge = oversample_extremes(n_train, n_layers, seed)

    df_main = enforce_physical_constraints(df_main, n_layers)
    df_edge = enforce_physical_constraints(df_edge, n_layers)

    df_train = pd.concat([df_main.iloc[:n_train], df_edge], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val   = df_main.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test  = df_main.iloc[n_train + n_val:].reset_index(drop=True)

    # run SNICAR
    runner = run_batch_parallel if n_jobs > 1 else run_batch

    bba_train = runner(df_train, n_layers, base_inputs, n_jobs, desc='Train') \
        if n_jobs > 1 else run_batch(df_train, n_layers, base_inputs, desc='Train')
    bba_val   = run_batch(df_val,   n_layers, base_inputs, desc='Val  ')
    bba_test  = run_batch(df_test,  n_layers, base_inputs, desc='Test ')

    # save
    print('\nSaving...')
    save_split(df_train, bba_train, output_dir / 'snicar_train.npz')
    save_split(df_val,   bba_val,   output_dir / 'snicar_val.npz')
    save_split(df_test,  bba_test,  output_dir / 'snicar_test.npz')

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train',    type=int, default=100_000)
    parser.add_argument('--n_val',      type=int, default=10_000)
    parser.add_argument('--n_test',     type=int, default=10_000)
    parser.add_argument('--n_layers',   type=int, default=4)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='snicar_data')
    parser.add_argument('--n_jobs',     type=int, default=1,
                        help='Parallel workers for SNICAR runs (default: 1)')
    args = parser.parse_args()

    main(
        n_train    = args.n_train,
        n_val      = args.n_val,
        n_test     = args.n_test,
        n_layers   = args.n_layers,
        seed       = args.seed,
        output_dir = args.output_dir,
        n_jobs     = args.n_jobs,
    )
