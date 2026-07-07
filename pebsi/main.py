"""
PEBSI main function

Defines the top-level @jax.jit compiled
execution loop for PEBSI. Each function takes
in the glacier state and returns an updated
glacier state.

Executed in the following order:
  1. Adds new mass (accumulation; particle deposition)
  2. Updates surface updates properties daily (albedo; days since snowfall)
  3. Solves surface energy balance equation for surface temperature
  4. Runs vertical proceses (melting; percolation; refreezing)
  5. Updates state properties hourly (densification; grain size)
  6. Runs annual routine to convert snow to firn
  7. Tracks mass error accumulated since previous timestep
"""
# Internal libraries
from types import SimpleNamespace
# External libraries
import jax 
import jax.numpy as jnp
# Local libraries
from pebsi.energybalance import EnergyBalanceDriver
from pebsi.massbalance import MassBalanceDriver
from pebsi.forcing import domain_expansion
from util.layers import *
from pebsi.state import StepOutputs

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
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})

    # initiate drivers
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def fetch_current_mass(state):
        """Fetches total current mass per-point [kg m-2]"""
        total_mass = jnp.sum(state.lice, axis=1) + \
                        jnp.sum(state.lwater, axis=1) + \
                        state.basal_reservoir
        return total_mass
    
    def mass_conservation(initial_mass, final_mass, mass_fluxes):
        """Calculates mass deviation from previous timestep [kg m-2]"""
        mass_in = mass_fluxes['accumulation'] + mass_fluxes['rainfall'] + \
            mass_fluxes['deposition'] + mass_fluxes['condensation']
        mass_out = mass_fluxes['runoff'] + mass_fluxes['evaporation'] + \
            mass_fluxes['sublimation'] + mass_fluxes['dead']
        
        return (initial_mass - final_mass) + (mass_in - mass_out)
    
    def step(current_state, current_forcings):
        """
        Runs all processes for a single timestep and
        updates the records.
        """
        # expand forcings from grid cell to simulation points
        current_forcings = domain_expansion(
            current_forcings, point_attrs, params
        )

        # initialize mass balance check
        current_mass = fetch_current_mass(current_state)
        
        # ===================== STEP 1 =====================
        # get amounts of rain and snow; add dry deposition
        rainfall, snowfall, current_state = mb.run_new_mass(
            current_state, current_forcings
        )

        # ===================== STEP 2 =====================
        # surface property updates: albedo, surrounding albedo
        current_state = mb.run_daily_routines(
            current_state, current_forcings
        )

        # ===================== STEP 3 =====================
        # solve energy balance equation for surface temperature
        current_state, fluxes = eb.solve_energy_balance(
            current_state, current_forcings, point_attrs
        )
        
        # ===================== STEP 3 =====================
        #          vertical heat and mass exchange
        fluxes_to_vert = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }
        current_state, mass_fluxes = mb.run_vertical_processes(
            current_state, current_forcings, fluxes_to_vert
        )

        # ===================== STEP 5 =====================
        #   evolution of density, grain size, and roughness
        current_state, water_squeezed_out = mb.run_state_updates(
            current_state, current_forcings
        )
        mass_fluxes['runoff'] = mass_fluxes['runoff'] + water_squeezed_out
        mass_fluxes['melt'] = jnp.sum(mass_fluxes['melt_2D'], axis=1)
        mass_fluxes['accumulation'] = snowfall
        mass_fluxes['rainfall'] = rainfall
        mass_fluxes['refreeze'] = jnp.sum(current_state.ldrefreeze, axis=1)
        mass_fluxes['cumrefreeze'] = jnp.sum(current_state.lrefreeze, axis=1)

        # ===================== STEP 6 =====================
        #          annual checks and tracker updates 
        current_state = mb.run_annual_routines(current_state, current_forcings)

        # ===================== STEP 7 =====================
        #            mass conservation tracking
        next_mass = fetch_current_mass(current_state)
        mass_fluxes['error'] = mass_conservation(
            current_mass, next_mass, mass_fluxes
        )

        # ===================== OUTPUTS =====================
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
                out[field] = mass_fluxes[field] / params.density_water

        # get all the layer fields
        for field in StepOutputs._fields:
            if field.startswith('layer'):
                state_field = field.replace('layer', 'l')
                values = getattr(next_state, state_field)

                if field in ['layerBC','layerOC','layerdust']:
                    lheight = next_state.lheight
                    safe_height = jnp.where(lheight > 0, lheight, 1e-6)
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
        step, initial_state, all_forcings, unroll=1
    )

    # ===== COMPLETED SIMULATION: STORE DATA =====
    return final_state, records