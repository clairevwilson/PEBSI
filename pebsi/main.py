# External libraries
import jax 
from pebsi.state import StepOutputs
import jax.numpy as jnp
from jax.debug import print as jax_print

# Local libraries
from pebsi.energybalance import EnergyBalanceDriver
from pebsi.massbalance import MassBalanceDriver
import pebsi.surface as surface
from util.layers import *

@jax.jit(static_argnames=['static_args'])
def main(
    initial_state, 
    all_forcings, 
    point_attrs, 
    static_args, 
    dynamic_args):
    """
    Core function which executes the time loop
    for all mass balance and energy balance 
    calculations.
    """
    # initiate drivers
    mb = MassBalanceDriver(static_args, dynamic_args)
    eb = EnergyBalanceDriver(static_args, dynamic_args)

    def fetch_current_mass(state):
        total_mass = jnp.sum(state.lice, axis=1) + \
                        jnp.sum(state.lwater, axis=1) + \
                        state.basal_reservoir
        return total_mass
    
    def mass_conservation(initial_mass, final_mass, mass_fluxes):
        mass_in = mass_fluxes['accumulation'] + mass_fluxes['rainfall'] + \
            mass_fluxes['deposition'] + mass_fluxes['condensation']
        mass_out = mass_fluxes['runoff'] + mass_fluxes['evaporation'] + \
            mass_fluxes['sublimation'] + mass_fluxes['dead']
        
        return (initial_mass - final_mass) + (mass_in - mass_out)
    
    # define function for a single timestep
    def step(current_state, current_forcings):
        # initialize mass balance check
        current_mass = fetch_current_mass(current_state)

        # 1. get amounts of rain and snow; add dry deposition
        rainfall, snowfall, current_state = mb.run_new_mass(
            current_state, current_forcings
        )
        all_mass_fluxes = ['rainfall','accumulation','deposition','condensation',
                           'sublimation','evaporation','runoff','dead']
        mf = {'accumulation': snowfall}
        for flux in all_mass_fluxes:
            if flux not in mf:
                mf[flux] = jnp.zeros_like(rainfall)
        # jax_print('1. {}', jnp.abs(mass_conservation(current_mass, fetch_current_mass(current_state), mf)) > 1e-3)

        # 2. surface property updates
        current_state = mb.run_daily_routines(
            current_state, current_forcings
        )
        mf = {'accumulation': snowfall}
        for flux in all_mass_fluxes:
            if flux not in mf:
                mf[flux] = jnp.zeros_like(rainfall)
        # jax_print('2. {}', jnp.abs(mass_conservation(current_mass, fetch_current_mass(current_state), mf)))

        # 3. simultaneously solve energy balance and surface temperature
        current_state, fluxes = eb.solve_energy_balance(
            current_state, current_forcings, point_attrs
        )

        # 4. vertical heat and mass exchange
        fluxes_to_vert = {
            'rainfall': rainfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }
        current_state, mass_fluxes = mb.run_vertical_processes(
            current_state, current_forcings, fluxes_to_vert
        )

        mf = {'accumulation': snowfall, 'rainfall': rainfall}
        for flux in all_mass_fluxes:
            if flux not in mf:
                if flux in mass_fluxes:
                    mf[flux] = mass_fluxes[flux]
                else:
                    mf[flux] = jnp.zeros_like(rainfall)
        # jax_print('4. {}', mass_conservation(current_mass, fetch_current_mass(current_state), mf))

        # 5. state property updates: density, grain size, surface roughness
        current_state, water_squeezed_out = mb.run_state_updates(
            current_state, current_forcings
        )
        mass_fluxes['runoff'] = mass_fluxes['runoff'] + water_squeezed_out
        mass_fluxes['melt'] = jnp.sum(mass_fluxes['melt_2D'], axis=1)
        mass_fluxes['accumulation'] = snowfall
        mass_fluxes['rainfall'] = rainfall
        mass_fluxes['refreeze'] = jnp.sum(current_state.ldrefreeze, axis=1)
        mass_fluxes['cumrefreeze'] = jnp.sum(current_state.lrefreeze, axis=1)

        for flux in all_mass_fluxes:
            if flux not in mf:
                if flux in mass_fluxes:
                    mf[flux] = mass_fluxes[flux]
                else:
                    mf[flux] = jnp.zeros_like(rainfall)
        # jax_print('5. {}', mass_conservation(current_mass, fetch_current_mass(current_state), mf))

        # 6. annual checks and tracker updates 
        current_state = mb.run_annual_routines(current_state, current_forcings)

        for flux in all_mass_fluxes:
            if flux not in mf:
                if flux in mass_fluxes:
                    mf[flux] = mass_fluxes[flux]
                else:
                    mf[flux] = jnp.zeros_like(rainfall)
        # jax_print('6. {}', mass_conservation(current_mass, fetch_current_mass(current_state), mf))

        # 7. mass conservation check
        next_mass = fetch_current_mass(current_state)
        mass_fluxes['error'] = mass_conservation(
            current_mass, next_mass, mass_fluxes
        )
        # jax_print('doy {} hour {}    actual error accumulated: {} {}', 
        #           current_forcings.doy, current_forcings.hour,
        #           jnp.abs(mass_fluxes['error']) > 1e-3, 
        #           mass_fluxes['error'])

        # define the next state
        next_state = current_state

        # pack climate outputs manually
        out = {
            'airtemp': current_forcings.tempC, 
            'rh': current_forcings.rh,
            'wind': current_forcings.wind,
            'winddir': current_forcings.winddir,
            'tp': current_forcings.tp,
            'sp': current_forcings.sp,
            'albedo': current_state.albedo
        }

        # get all the fields from energy balance fluxes
        for field in fluxes:
            if field in StepOutputs._fields:
                out[field] = fluxes[field]

        # get all the fields from mass fluxes
        for field in mass_fluxes:
            if field in StepOutputs._fields:
                # store them in m w.e.
                out[field] = mass_fluxes[field] / static_args.density_ice

        # get all the layer fields
        for field in StepOutputs._fields:
            if field.startswith('layer'):
                state_field = field.replace('layer', 'l')
                values = getattr(next_state, state_field)

                if field in ['layerBC','layerOC','layerdust']:
                    lheight = next_state.lheight
                    safe_height = jnp.where(lheight>0, lheight, 1e-6)
                    concentration = values / safe_height

                    # put into interpretable units (ppb / ppm)
                    if field in ['layerBC','layerOC']:
                        values = concentration * 1e6
                    else:
                        values = concentration * 1e3

                # store the values to output dictionary
                out[field] = values

        # store records
        step_records = StepOutputs(**out)
        return next_state, step_records
    
    # execute model with jax.lax.scan
    final_state, records = jax.lax.scan(
        step, initial_state, all_forcings
    )

    # ===== COMPLETED SIMULATION: STORE DATA =====
    return final_state, records