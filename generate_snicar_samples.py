"""
SNICAR Emulator - Sample Generation
=====================================
Generates Latin Hypercube Sampling (LHS) samples and runs SNICAR
for each, storing inputs and broadband albedo outputs for emulator training.

Discrete parameters (direct illumination, grain shape) are sampled
continuously in [0, 1] by LHS, then snapped to their valid integer
values after sampling.

Outputs:
    snicar_train.npz: training set (LHS + extreme oversamples, shuffled)
    snicar_val.npz:   validation set (LHS only)
    snicar_test.npz:  test set (LHS only)
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

# Define input file paths
YAML_PATH = '/Users/cvw/local/PEBSI/biosnicar-py/biosnicar/inputs.yaml'
SFC_PATH  = 'Data/OP_data/480band/r_sfc/gulkana_cleanice_avg_bba3732.csv'

# Dust bin mass fractions: splits total dust concentration into SNICAR size bins
BIN1 = 0.0751       # 0.05–0.5 µm
BIN2 = 0.20535      # 0.5–1.25 µm
BIN3 = 0.481675     # 1.25–2.5 µm
BIN4 = 0.203775     # 2.5–5 µm
BIN5 = 0.034        # 5–50 µm

# ICE sub-fields that are copied uniformly across all layers from the base YAML
ICE_CONSTANTS = ['LAYER_TYPE', 'HEX_SIDE', 'HEX_LENGTH', 'SHP_FCTR', 'WATER_COATING', 'CDOM']

# Valid grain shape codes and their corresponding aspect ratios.
# 0 = spherical grains  (aspect ratio unused, set to 0)
# 2 = hexagonal plates  (aspect ratio = 0.01)
GRAIN_SHAPE_AR = {0: 0.0, 2: 0.01}

# Continuous sampling bounds for each physical parameter (min, max).
# Discrete params (grain_shape, direct) are sampled in [0, 1] then snapped
# to their valid values by snap_discrete_params().
PARAM_RANGES = {
    'grain_size_um':    (55.0,    1500.0),   # capped at the limits PEBSI uses
    'density_kgm3':     (50.0,    700.0),
    'height_m':         (0.01,    0.5),      # per-layer; total column capped at 1.0 m
    'bc_ppb':           (0.0,     5000.0),   # set max at an extremely high value
    'oc_ppb':           (0.0,     5000.0),
    'dust_ppb':         (0.0,     500000.0),
    'grain_shape':      (0.0,     1.0),      # snapped: 0 (spherical) or 2 (hex plates)
    'solar_zenith_deg': (0.0,     80.0),
    'direct':           (0.0,     1.0),      # snapped: 0 (diffuse) or 1 (direct beam)
}

# Layer-level parameters (one value per layer per sample)
LAYER_PARAMS  = ['grain_size_um', 'density_kgm3', 'height_m',
                 'bc_ppb', 'oc_ppb', 'dust_ppb', 'grain_shape']
# Column-level parameters (one value per sample)
COLUMN_PARAMS = ['solar_zenith_deg', 'direct']

# ============================================================
# SAMPLING
# ============================================================

def build_column_names(n_layers):
    """Returns the ordered list of feature column names for a given layer count."""
    cols = []
    for i in range(n_layers):
        for param in LAYER_PARAMS:
            cols.append(f'layer{i}_{param}')
    return cols + COLUMN_PARAMS


def snap_discrete_params(df):
    """
    Snaps continuously-sampled discrete parameters to their valid integer values.

      direct:      [0, 1] -> round  -> 0 (diffuse) or 1 (direct beam)
      grain_shape: [0, 1] -> round, *2 -> 0 (spherical) or 2 (hexagonal plates)
    """
    # snap illumination type: round to nearest integer (0 or 1)
    if 'direct' in df.columns:
        df['direct'] = df['direct'].round().astype(int)

    # snap grain shape: round [0,1] -> {0,1}, then *2 -> {0,2}
    shape_cols = [c for c in df.columns if c.endswith('_grain_shape')]
    for col in shape_cols:
        df[col] = (df[col].round() * 2).astype(int)

    return df


def lhs_samples(n_samples, n_layers, seed):
    """
    Draws n_samples via Latin Hypercube Sampling over all parameters,
    then snaps discrete parameters to their valid values.
    """
    cols = build_column_names(n_layers)
    lo, hi = [], []

    for col in cols:
        # strip 'layerN_' prefix to get the base parameter name
        key = col.split('_', 1)[1] if col.startswith('layer') else col
        lo.append(PARAM_RANGES[key][0])
        hi.append(PARAM_RANGES[key][1])

    # sample parameter space scaled to the valid ranges
    sampler = qmc.LatinHypercube(d=len(cols), seed=seed)
    scaled = qmc.scale(sampler.random(n=n_samples), lo, hi)
    df = pd.DataFrame(scaled, columns=cols)
    return snap_discrete_params(df)


def oversample_extremes(n_samples, n_layers, seed):
    """
    Generates n_samples // 5 hand-crafted extreme-regime samples to improve
    emulator coverage at the tails of the input distribution.

    Scenarios:
      0 - very clean, fine-grained ice
      1 - very dirty, coarse-grained ice
      2 - large grains, moderate impurities
      3 - thin surface layer over deeper ice
      4 - high solar zenith angle
    """
    rng     = np.random.default_rng(seed + 999)
    n_edge  = n_samples // 5
    cols    = build_column_names(n_layers)
    records = []

    for _ in range(n_edge):
        row      = {}
        scenario = rng.integers(0, 5)

        for i in range(n_layers):
            # randomly sample density so it increases with depth, capped to density max
            row[f'layer{i}_density_kgm3'] = float(min(rng.uniform(50, 700) * (1 + i * 0.1), 700))
            # randomly assign height in 1 - 50 cm
            row[f'layer{i}_height_m']     = float(rng.uniform(0.01, 0.50))
            # randomly assign spherical (0) or hexagonal plate (2) grain shape
            row[f'layer{i}_grain_shape']  = int(rng.choice([0, 2]))

            if scenario == 0:   # very clean, fine-grained
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(50, 200))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 10))
                row[f'layer{i}_oc_ppb']        = float(rng.uniform(0, 10))
                row[f'layer{i}_dust_ppb']      = float(rng.uniform(0, 50))
            elif scenario == 1: # very dirty, coarse-grained
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(500, 2000))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(2000, 5000))
                row[f'layer{i}_oc_ppb']        = float(rng.uniform(1000, 5000))
                row[f'layer{i}_dust_ppb']      = float(rng.uniform(5000, 10000))
            elif scenario == 2: # large grains, moderate impurities
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(1000, 2000))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 500))
                row[f'layer{i}_oc_ppb']        = float(rng.uniform(0, 500))
                row[f'layer{i}_dust_ppb']      = float(rng.uniform(0, 1000))
            elif scenario == 3: # thin surface layer over deeper ice
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(500, 1500))
                row[f'layer{i}_height_m']      = float(rng.uniform(0.005, 0.02) if i == 0 else rng.uniform(0.01, 0.50))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 1000))
                row[f'layer{i}_oc_ppb']        = float(rng.uniform(0, 1000))
                row[f'layer{i}_dust_ppb']      = float(rng.uniform(0, 2000))
            else:               # high solar zenith angle
                row[f'layer{i}_grain_size_um'] = float(rng.uniform(50, 500))
                row[f'layer{i}_bc_ppb']        = float(rng.uniform(0, 200))
                row[f'layer{i}_oc_ppb']        = float(rng.uniform(0, 200))
                row[f'layer{i}_dust_ppb']      = float(rng.uniform(0, 500))

        # assign high solar zenith angle unless using scenario 4
        row['solar_zenith_deg'] = float(rng.uniform(70, 85) if scenario == 4 else rng.uniform(0, 85))
        # randomly assign direct (1) or diffuse (0) illumination
        row['direct'] = int(rng.choice([0, 1]))
        records.append(row)

    return pd.DataFrame(records, columns=cols)


def enforce_physical_constraints(df, n_layers):
    """
    Applies physical consistency rules before running SNICAR:
      - Density increases monotonically with depth.
      - Total column depth is capped at 1.0 m by proportionally rescaling all
        layer thicknesses.
    """
    for i in range(1, n_layers):
        # deeper layers must not be substantially lighter or finer than the layer above
        df[f'layer{i}_density_kgm3']  = np.maximum(df[f'layer{i}_density_kgm3'],  df[f'layer{i-1}_density_kgm3']  * 0.85)

    # rescale layer thicknesses so total column depth does not exceed 1.0 m
    height_cols = [f'layer{i}_height_m' for i in range(n_layers)]
    scale = np.minimum(1.0, 1.0 / df[height_cols].sum(axis=1))
    for col in height_cols:
        df[col] *= scale

    return df


# ============================================================
# SNICAR RUNNER
# ============================================================

def load_base_inputs():
    """Loads the SNICAR YAML configuration as a dict."""
    with open(YAML_PATH, 'r') as f:
        return yaml.safe_load(f)


def run_snicar(row, n_layers, base_inputs):
    """
    Configures and runs SNICAR for a single sample row.
    Returns broadband albedo (float), or NaN if SNICAR raises an exception
    or the result falls outside [0, 1].

    Grain shape codes:
      0 -> spherical grains  (aspect ratio = 0, AR field unused)
      2 -> hexagonal plates  (aspect ratio = 0.01)

    Illumination:
      direct=1 -> direct solar beam, angle set by solar_zenith_deg
      direct=0 -> isotropic diffuse sky radiation
    """
    inputs = copy.deepcopy(base_inputs)

    # --- per-layer physical properties ---
    lheight = [float(row[f'layer{i}_height_m']) for i in range(n_layers)]
    ldensity = [int(row[f'layer{i}_density_kgm3']) for i in range(n_layers)]
    lgrainsize = [int(row[f'layer{i}_grain_size_um']) for i in range(n_layers)]
    lBC = [float(row[f'layer{i}_bc_ppb']) for i in range(n_layers)]
    lOC = [float(row[f'layer{i}_oc_ppb']) for i in range(n_layers)]
    ldust = np.array([float(row[f'layer{i}_dust_ppb']) for i in range(n_layers)])
    lshape = [int(row[f'layer{i}_grain_shape']) for i in range(n_layers)]

    # snap grain sizes to the SNICAR lookup grid (1 µm steps up to 1500, then 500 µm steps)
    lgrainsize = [
        int(np.round(r / 500) * 500) if r > 1500 else r
        for r in lgrainsize
    ]

    # aspect ratio: 0 for spherical, 0.01 for hexagonal plates
    lar = [GRAIN_SHAPE_AR[s] for s in lshape]

    # pack variables into inputs dict
    inputs['ICE']['DZ']  = lheight
    inputs['ICE']['RHO'] = ldensity
    inputs['ICE']['RDS'] = lgrainsize
    inputs['ICE']['LWC'] = [0] * n_layers
    inputs['ICE']['SHP'] = lshape
    inputs['ICE']['AR']  = lar

    # layer type is always 0 (granular snow)
    inputs['ICE']['LAYER_TYPE'][0] = 0

    # propagate scalar ICE constants uniformly across all layers
    for var in ICE_CONSTANTS:
        inputs['ICE'][var] = [inputs['ICE'][var][0]] * n_layers

    inputs['IMPURITIES']['BC']['CONC'] = lBC
    inputs['IMPURITIES']['OC']['CONC'] = lOC
    inputs['IMPURITIES']['DUST1']['CONC'] = (ldust * BIN1).tolist()
    inputs['IMPURITIES']['DUST2']['CONC'] = (ldust * BIN2).tolist()
    inputs['IMPURITIES']['DUST3']['CONC'] = (ldust * BIN3).tolist()
    inputs['IMPURITIES']['DUST4']['CONC'] = (ldust * BIN4).tolist()
    inputs['IMPURITIES']['DUST5']['CONC'] = (ldust * BIN5).tolist()

    inputs['PATHS']['SFC'] = SFC_PATH
    inputs['RTM']['SOLZEN'] = int(row['solar_zenith_deg'])
    inputs['RTM']['DIRECT'] = int(row['direct']) # 0 = diffuse, 1 = direct beam

    # run SNICAR using inputs and calculate broadband albedo from spectral weights
    try:
        albedo, spectral_weights = get(inputs)
        bba = float(np.sum(albedo * spectral_weights) / np.sum(spectral_weights))
        if not (0.0 <= bba <= 1.0):
            return np.nan
        return bba
    except Exception as e:
        print(f"FAILED: {e}")
        return np.nan


def run_batch(df, n_layers, base_inputs, desc='Running SNICAR'):
    """Runs SNICAR serially for every row in df; returns array of broadband albedos."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        results.append(run_snicar(row, n_layers, base_inputs))
    return np.array(results, dtype=np.float32)


def run_batch_parallel(df, n_layers, base_inputs, n_jobs, desc='Running SNICAR'):
    """Runs SNICAR in parallel via joblib; returns array of broadband albedos."""
    rows = [row for _, row in df.iterrows()]
    results = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
        joblib.delayed(run_snicar)(row, n_layers, base_inputs)
        for row in tqdm(rows, desc=desc)
    )
    return np.array(results, dtype=np.float32)


# ============================================================
# SAVE
# ============================================================

def save_split(df, bba, path):
    """
    Saves inputs (X) and broadband albedo targets (y) to a .npz file.
    Failed SNICAR runs (NaN bba) are silently dropped before saving.
    X shape: (N, n_features)  |  y shape: (N,)
    """
    valid    = ~np.isnan(bba)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f'  Warning: {n_dropped} failed SNICAR runs dropped from {path.name}')

    X = df.values[valid].astype(np.float32)
    y = bba[valid]

    np.savez(path, X=X, y=y, columns=np.array(df.columns.tolist()))
    print(f'  Saved {path}  ({valid.sum():,} samples, {X.shape[1]} features)')


# ============================================================
# MAIN
# ============================================================

def main(n_train, n_val, n_test, n_layers, seed, output_dir, n_jobs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = n_layers * len(LAYER_PARAMS) + len(COLUMN_PARAMS)
    print(f'\n=== SNICAR Emulator Sample Generation ===')
    print(f'Layers: {n_layers}  |  Features: {n_features}')
    print(f'Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}')
    print(f'Jobs: {n_jobs}\n')

    base_inputs = load_base_inputs()
    total = n_train + n_val + n_test

    # --- sample generation ---
    print('Sampling...')
    df_main = lhs_samples(total, n_layers, seed)
    df_edge = oversample_extremes(n_train, n_layers, seed)

    # enforce physical plausibility before passing to SNICAR
    df_main = enforce_physical_constraints(df_main, n_layers)
    df_edge = enforce_physical_constraints(df_edge, n_layers)

    # training set: LHS core + extreme oversamples, shuffled together
    df_train = pd.concat([df_main.iloc[:n_train], df_edge], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val   = df_main.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test  = df_main.iloc[n_train + n_val:].reset_index(drop=True)

    # --- run SNICAR ---
    runner = run_batch_parallel if n_jobs > 1 else run_batch

    bba_train = runner(df_train, n_layers, base_inputs, n_jobs, desc='Train') \
        if n_jobs > 1 else run_batch(df_train, n_layers, base_inputs, desc='Train')
    bba_val   = run_batch(df_val,   n_layers, base_inputs, desc='Val  ')
    bba_test  = run_batch(df_test,  n_layers, base_inputs, desc='Test ')

    # --- save ---
    print('\nSaving...')
    save_split(df_train, bba_train, output_dir / 'snicar_train.npz')
    save_split(df_val, bba_val, output_dir / 'snicar_val.npz')
    save_split(df_test, bba_test, output_dir / 'snicar_test.npz')

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SNICAR training samples via LHS.')
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_val', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='snicar_data')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel workers for SNICAR runs (default: 1)')
    args = parser.parse_args()

    main(
        n_train = args.n_train,
        n_val = args.n_val,
        n_test = args.n_test,
        n_layers = args.n_layers,
        seed = args.seed,
        output_dir = args.output_dir,
        n_jobs = args.n_jobs,
    )
