"""
Surface class for PEBSI

Calculates the surface properties such
as albedo and surface temperature.

@author: clairevwilson
"""
# External libraries
import equinox as eqx
import numpy as np
import jax 
import jax.numpy as jnp

class SNICAREmulator(eqx.Module):
    layers: list
 
    def __init__(self, in_dim, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_dim, 128, key=keys[0]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,    128, key=keys[1]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,     64, key=keys[2]), eqx.nn.LayerNorm((64,)),
            eqx.nn.Linear(64,       1, key=keys[3]),
        ]
 
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x)) if isinstance(layer, eqx.nn.Linear) else layer(x)
        return jax.nn.sigmoid(self.layers[-1](x)).squeeze()
    
model = eqx.tree_deserialise_leaves(
    'snicar_emulator.eqx',
    eqx.tree_at(lambda m: m, SNICAREmulator(30, jax.random.PRNGKey(0)), 
                replace_fn=lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x)
)
norm  = np.load('snicar_norm.npz')
mu, sigma = jnp.array(norm['mu']), jnp.array(norm['sigma'])

def get_albedo(state, args, forcings):
    lheight = state.lheight[:, :4]
    ldensity = state.ldensity[:, :4]
    lgrainsize = state.lgrainsize[:, :4]
    lBC = state.lBC[:, :4]
    lOC = state.lOC[:, :4]
    ldust = state.ldust[:, :4]

    solar_zenith = forcings.solar_zenith 
    direct = (forcings.tcc <= args.diffuse_cloud_limit).astype(jnp.float32)

    round_grains = (state.lrefreeze > 0) | (state.lwater > 1e-3)
    lgrainshape = jnp.where(round_grains[:, :4], 0, 2).astype(jnp.float32)

    cBC = lBC / lheight * 1e6
    cOC = lOC / lheight * 1e6
    cdust = ldust / lheight * 1e6

    # stack inputs
    X = jnp.concatenate([
        jnp.stack([lgrainsize[:, i], ldensity[:, i], lheight[:, i],
                   cBC[:, i], cOC[:, i], cdust[:, i], lgrainshape[:, i]], axis=1)
        for i in range(4)
    ] + [solar_zenith[:, None], direct[:, None]], axis=1)
    
    # calculate albedo from emulator
    albedo = jax.vmap(model)((X - mu) / sigma)  # (N_POINTS,)

    # check if this is lower than the current year albedo
    year_idx = (forcings.year - args.start_year)
    new_annual_min_albedo = state.annual_min_albedo.at[:, year_idx].set(
        jnp.minimum(state.annual_min_albedo[:, year_idx], albedo)
    )

    # for exposed firn, get the minimum albedo from that year
    exposed_year = (state.lage[:, 0] / 365.25).astype(int)
    exposed_idx = exposed_year - args.start_year
    albedo_firn = new_annual_min_albedo[jnp.arange(state.lage.shape[0]), exposed_idx]
    # make sure we aren't filling with an initialized 1
    albedo_firn = jnp.where(
        albedo_firn < 1, albedo_firn, args.albedo_firn
    )

    final_albedo = jnp.where(
        state.ltype[:, 0] == 0,
        albedo,
        jnp.where(state.ltype[:, 0] == 1,
                  albedo_firn, args.albedo_ice)
    )
    return final_albedo, new_annual_min_albedo