"""
Albedo module for PEBSI

Wraps the BioSNICAR neural-network emulator (.npz format) in a
pure-JAX forward pass so it runs on GPU inside jit/vmap without
any host-side numpy calls at inference time.
"""
import json
import numpy as np
import jax.numpy as jnp
import jax
from pebsi.defaults import emulator_fn as _NPZ_PATH

# ── Load emulator weights once at import time ──────────────────────────────

def _load_emulator(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data['metadata']))

    weights, biases = [], []
    i = 0
    while f'weights_{i}' in data:
        weights.append(jnp.array(data[f'weights_{i}'], dtype=jnp.float32))
        biases.append(jnp.array(data[f'biases_{i}'], dtype=jnp.float32))
        i += 1

    param_names = meta['param_names']

    # input transforms and activation recorded at training time; older
    # emulators predate these fields and used linear scaling + ReLU
    input_transform = meta.get('input_transform', {})
    activation = meta.get('activation', 'relu')
    for name, t in input_transform.items():
        if t not in ('log1p', 'identity', 'none', None):
            raise ValueError(f"Unsupported input_transform {t!r} for {name!r}")

    # one flag per input column for the log1p-transformed params
    log1p_mask = jnp.array(
        [input_transform.get(name) == 'log1p' for name in param_names], dtype=bool)

    return {
        'weights':        weights,
        'biases':         biases,
        'pca_components': jnp.array(data['pca_components'], dtype=jnp.float32),
        'pca_mean':       jnp.array(data['pca_mean'],       dtype=jnp.float32),
        'input_min':      jnp.array(data['input_min'],      dtype=jnp.float32),
        'input_max':      jnp.array(data['input_max'],      dtype=jnp.float32),
        'flx_slr':        jnp.array(data['flx_slr'],        dtype=jnp.float32),
        'param_names':    param_names,
        'log1p_mask':     log1p_mask,
        'activation':     activation,
    }

_emu = _load_emulator(_NPZ_PATH)
_flx_slr_norm = _emu['flx_slr'] / _emu['flx_slr'].sum()  # normalized for BBA

# jax.nn versions are overflow-safe, so their derivatives stay finite
_ACTIVATIONS = {
    'relu':     jax.nn.relu,
    'tanh':     jnp.tanh,
    'logistic': jax.nn.sigmoid,
    'identity': lambda x: x,
    'softplus': jax.nn.softplus,
    'silu':     jax.nn.silu,
}
_activation_fn = _ACTIVATIONS[_emu['activation']]


def _transform_inputs(X):
    """Applies the recorded input transforms (log1p on flagged columns) to raw X."""
    return jnp.where(_emu['log1p_mask'], jnp.log1p(X), X)


def _forward(x):
    """Pure-JAX MLP + PCA reconstruction for a single input vector (n_params,)."""
    for i, (W, b) in enumerate(zip(_emu['weights'], _emu['biases'])):
        x = x @ W + b
        if i < len(_emu['weights']) - 1:
            x = _activation_fn(x)
    spectral = x @ _emu['pca_components'] + _emu['pca_mean']
    return jnp.clip(spectral, 0.0, 1.0)  # (480,)


def get_albedo(state, params, forcings):
    """
    Calculates albedo using the BioSNICAR emulator and tracks
    annual minimum albedo. When firn layers are exposed, the surface
    uses the minimum albedo from the year the firn was created.
    Ice has a constant albedo.

    Emulator parameters (top layer only), in trained order:
      rds               grain radius [um]
      rho               density [kg m-3]
      black_carbon      BC concentration [ppb by mass]
      brown_carbon      OC concentration [ppb by mass]
      dust_total        total dust concentration [ppb] — split into 5 bins internally
                        using PEBSI bin ratios (same as defaults.py)
      solzen            solar zenith angle [degrees]
      direct            1 if direct illumination, 0 if diffuse
      shp               grain shape: 0=sphere, 1=spheroid, 2=hex plate
    """

    rds = state.lgrainsize[:, 0]
    rho = state.ldensity[:, 0]

    # concentrations in ppb over a fixed near-surface depth rather than the
    # top layer alone: a melting top layer's mass can shrink toward zero,
    # and dividing by it made concentration -> albedo -> melt -> layer mass
    # a runaway loop in the gradients. Each layer counts by the fraction of
    # it lying within albedo_impurity_depth of the surface
    lheight = state.lheight
    layer_top = jnp.cumsum(lheight, axis=1) - lheight
    safe_height = jnp.where(lheight > 0, lheight, 1.0)
    frac_within = jnp.clip((params.albedo_impurity_depth - layer_top) / safe_height, 0.0, 1.0)
    window_mass = jnp.sum(lheight * state.ldensity * frac_within, axis=1)
    safe_mass = jnp.where(window_mass > 0, window_mass, 1.0)
    cBC = jnp.sum(state.lBC * frac_within, axis=1) / safe_mass * 1e9
    cOC = jnp.sum(state.lOC * frac_within, axis=1) / safe_mass * 1e9
    cdust = jnp.sum(state.ldust * frac_within, axis=1) / safe_mass * 1e9

    has_refreeze = state.lrefreeze[:, 0] > 1e-3
    has_water = state.lwater[:, 0] > 1e-3

    solzen = jnp.rad2deg(forcings.solar_zenith)
    direct = (forcings.tcc <= params.diffuse_cloud_limit).astype(jnp.float32)

    # hex plates for dry snow with no refreeze; spheres otherwise
    if params.option_flat_plates:
        dry = ~has_refreeze & ~has_water
        shp = jnp.where(dry, 2.0, 0.0)
    else:
        shp = jnp.full_like(cBC, 0.0)

    # assemble input in the order the emulator was trained on
    X = jnp.stack([rds, rho, cBC, cOC, cdust,
                   solzen, direct, shp], axis=1)  # (N, 8)

    # recorded input transforms first: input_min/input_max are in the
    # transformed space. Then min-max scale to [0, 1]
    X = _transform_inputs(X)
    X = (X - _emu['input_min']) / (_emu['input_max'] - _emu['input_min'] + 1e-30)
    X = jnp.clip(X, 0.0, 1.0)

    # run emulator for each grid point → spectral albedo (N, 480)
    spectral = jax.vmap(_forward)(X)

    # broadband albedo weighted by solar spectrum
    albedo = jnp.sum(spectral * _flx_slr_norm, axis=1)

    # track annual minimum albedo
    year_idx = (forcings.year - params.start_year)
    new_annual_min_albedo = state.annual_min_albedo.at[:, year_idx].set(
        jnp.minimum(state.annual_min_albedo[:, year_idx], albedo)
    )

    # for exposed firn, use the minimum albedo from the year it was exposed
    exposed_year = (state.lage[:, 0] / 365.25).astype(int)
    exposed_idx = exposed_year - params.start_year
    albedo_firn = new_annual_min_albedo[jnp.arange(state.lage.shape[0]), exposed_idx]

    albedo_firn = jnp.where(albedo_firn < 1, albedo_firn, params.albedo_firn)

    final_albedo = jnp.where(
        state.ltype[:, 0] == 0,
        albedo,
        jnp.where(state.ltype[:, 0] == 1, albedo_firn, params.albedo_ice)
    )
    return final_albedo, new_annual_min_albedo
