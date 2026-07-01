"""
Albedo class for PEBSI

Contains functions to load and apply the 
emulator for SNICAR (the Snow, Ice and Aerosol
Radiative model)
"""
# External libraries
import equinox as eqx
import numpy as np
import jax 
import jax.numpy as jnp

class SNICAREmulator(eqx.Module):
    """
    Small MLP emulator for SNICAR broadband albedo.

    Architecture: Linear -> LayerNorm -> GELU x3
                  then Linear -> Sigmoid

    Takes in a flat vector of snowpack properties and
    returns broadband albedo bounded to [0, 1].
    """
    layers: list
 
    def __init__(self, in_dim, key):
        # generate random key and split it into 4 independent subkeys for layers
        keys = jax.random.split(key, 4)

        # build sequential training layers
        self.layers = [
            eqx.nn.Linear(in_dim, 128, key=keys[0]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128, 128, key=keys[1]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128, 64, key=keys[2]), eqx.nn.LayerNorm((64,)),
            eqx.nn.Linear(64, 1, key=keys[3]),
        ]
 
    def __call__(self, x):
        # wrap Linear layers in GELU activation; apply LayerNorm without activation
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x)) if isinstance(layer, eqx.nn.Linear) else layer(x)
        return jax.nn.sigmoid(self.layers[-1](x)).squeeze()

# load the model weights and create the template from custom SNICAR class
model = eqx.tree_deserialise_leaves(
    'snicar_emulator/emulator.eqx',
    eqx.tree_at(lambda m: m, SNICAREmulator(30, jax.random.PRNGKey(0)), 
                replace_fn=lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x)
)

# load the normalization weights
norm = np.load('snicar_emulator/normalization.npz')
mu, sigma = jnp.array(norm['mu']), jnp.array(norm['sigma'])

def get_albedo(state, params, forcings):
    """
    Calculates albedo using the emulator and tracks
    annual minimum albedo. When firn layers are 
    exposed, the surface uses the minimum albedo from 
    the year the firn was created. Ice has a constant
    albedo.
    """
    # grab the top four layers
    lheight = state.lheight[:, :4]
    ldensity = state.ldensity[:, :4]
    lgrainsize = state.lgrainsize[:, :4]
    lBC = state.lBC[:, :4]
    lOC = state.lOC[:, :4]
    ldust = state.ldust[:, :4]

    # grab 1D inputs
    solar_zenith = jnp.rad2deg(forcings.solar_zenith)
    direct = (forcings.tcc <= params.diffuse_cloud_limit).astype(jnp.float32)

    if params.option_flat_plates:
        # spherical grains if there is refreeze or liquid water; else flat hexagons
        round_grains = (state.lrefreeze > 0) | (state.lwater > 1e-3)
        lgrainshape = jnp.where(round_grains[:, :4], 0, 2).astype(jnp.float32)
    else:
        lgrainshape = jnp.full_like(lheight, 0).astype(jnp.float32)

    # calculate concentration from mass of particles and convert to ppb
    cBC = lBC / lheight * 1e6
    cOC = lOC / lheight * 1e6
    cdust = ldust / lheight * 1e6

    # truncate layer height so it doesn't exceed the 1.0 m the emulator was trained on
    cumulative_height = jnp.cumsum(lheight, axis=1)
    overshoot = jnp.maximum(0.0, cumulative_height - 1.0)
    lheight = jnp.maximum(0.0, lheight - overshoot)

    # stack inputs
    X = jnp.concatenate([
        jnp.stack([lgrainsize[:, i], ldensity[:, i], lheight[:, i],
                   cBC[:, i], cOC[:, i], cdust[:, i], lgrainshape[:, i]], axis=1)
        for i in range(4)] + [solar_zenith[:, None], direct[:, None]], axis=1)
    
    # apply weights to the input
    X_weighted = (X - mu) / sigma

    # calculate albedo using emulator
    albedo = jax.vmap(model)(X_weighted)

    # check if this is lower than the current year albedo
    year_idx = (forcings.year - params.start_year)
    new_annual_min_albedo = state.annual_min_albedo.at[:, year_idx].set(
        jnp.minimum(state.annual_min_albedo[:, year_idx], albedo)
    )

    # for exposed firn, get the minimum albedo from the year exposed
    exposed_year = (state.lage[:, 0] / 365.25).astype(int)
    exposed_idx = exposed_year - params.start_year
    albedo_firn = new_annual_min_albedo[jnp.arange(state.lage.shape[0]), exposed_idx]

    # fill initialized 1 values with our constant albedo_firn
    albedo_firn = jnp.where(
        albedo_firn < 1, albedo_firn, params.albedo_firn
    )

    final_albedo = jnp.where(
        state.ltype[:, 0] == 0,
        albedo,
        jnp.where(state.ltype[:, 0] == 1,
                  albedo_firn, params.albedo_ice)
    )
    return final_albedo, new_annual_min_albedo