# External libraries
import jax 
from pebsi.state import StepOutputs
import jax.numpy as jnp

# Local libraries
from pebsi.energybalance import EnergyBalanceDriver
from pebsi.massbalance import MassBalanceDriver
import pebsi.surface as surface
from util.layers import *

@jax.jit(static_argnames=['args'])
def main(initial_state, all_forcings, point_attrs, args):
    """
    Core function which executes the time loop
    for all mass balance and energy balance 
    calculations.
    """
    # initiate drivers
    mb = MassBalanceDriver(None, args)
    eb = EnergyBalanceDriver(None, args)

    # define function for a single timestep
    def step(current_state, current_forcings):
        time_idx = current_forcings.time_idx

        # initialize mass balance check
        current_mass = jnp.sum(current_state.lice, axis=1) + \
            jnp.sum(current_state.lwater, axis=1) + current_state.basal_reservoir

        # 1. get amounts of rain and snow; add dry deposition
        rainfall, snowfall, current_state = mb.add_new_mass(
            current_state, current_forcings
        )

        # 2. surface property updates
        current_state = mb.run_daily_routines(
            current_state, current_forcings, point_attrs
        )

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
        current_state, mass_fluxes = mb.vertical_processes(
            current_state, current_forcings, point_attrs, fluxes_to_vert
        )

        # 5. state property updates: density, grain size, surface roughness
        current_state, water_squeezed_out = mb.state_updates(
            current_state, current_forcings
        )
        mass_fluxes['runoff'] = mass_fluxes['runoff'] + water_squeezed_out
        mass_fluxes['melt'] = jnp.sum(mass_fluxes['melt_2D'], axis=1)
        mass_fluxes['accumulation'] = snowfall
        mass_fluxes['rainfall'] = rainfall
        mass_fluxes['refreeze'] = jnp.sum(current_state.ldrefreeze, axis=1)
        mass_fluxes['cumrefreeze'] = jnp.sum(current_state.lrefreeze, axis=1)

        # 6. annual checks and tracker updates 
        current_state = mb.run_annual_routines(current_state, current_forcings)

        # 7. mass conservation check
        mass_in = mass_fluxes['accumulation'] + mass_fluxes['rainfall'] + \
            mass_fluxes['deposition'] + mass_fluxes['condensation']
        mass_out = mass_fluxes['runoff'] + mass_fluxes['evaporation'] + \
            mass_fluxes['sublimation']
        next_mass = jnp.sum(current_state.lice, axis=1) + \
            jnp.sum(current_state.lwater, axis=1) + current_state.basal_reservoir
        mass_fluxes['error'] = (current_mass - next_mass) + (mass_in - mass_out)

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
                out[field] = mass_fluxes[field] / args.density_ice

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