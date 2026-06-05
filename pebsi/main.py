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
        fluxes_to_vert = {'rainfall': rainfall,
            'melt_heat': fluxes['melt_heat'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }
        current_state = mb.vertical_processes(
            current_state, current_forcings, point_attrs, fluxes_to_vert
        )

        # ACTUAL ORDER TO IMPLMEMENT:
        # 1: add mass (accumulation, dry deposition)
        # 2: solve energy balance equation
        # 3: vertical heat/mass exchange 
        #   (melt->phase changes-> temp profile->percolation->refreezing)
        # 4: densification
        # 5: layer management and trackers 
        # 6: mass conservation

        # # 5. melting: melt mass due to surface melt energy and penetrating shortwave
        # mb.melting() # ** do this a lot better -- handle subsurface, melted layers and surface in one

        # # 6. percolation: route meltwater and LAPs through snow/firn
        # runoff = mb.percolation(layermelt,rainfall)
        
        # # 7. temperature profile: resolve thermal conduction in vertical columns
        # mb.thermal_conduction()

        # # 8. phase changes: sublimation, evaporation, etc.
        # mb.phase_changes()

        # # 9. refreezing
        # mb.refreezing()

        # # 10. densification (only done daily)
        # current_state = jax.lax.cond(
        #     is_day_start or is_albedo_step,
        #     run_daily_routines,
        #     skip_daily_routines,
        #     current_state
        # )

        # 11. check all the trackers and layer sizes
        # layers.check_layer_sizes
        # # if towards the end of summer, check if old snow should become firn
        # doy = time.day_of_year
        # date_in_range = doy >= args.start_end_summer and doy <= args.start_end_summer + 60
        # if date_in_range and time.hour == 0 and not self.firn_converted:
        #     self.end_of_summer()

        # # if start of calendar year, reset annual trackers
        # if time.day_of_year == 1 and time.hour == 0:
        #     self.firn_converted = False
        #     self.ice_exposed = False

        # 12. mass conservation check
        # mass_in = snowfall + rainfall + self.condensation + self.deposition
        # mass_out = runoff + self.evaporation + self.sublimation
        # self.check_mass_conservation(mass_in, mass_out)
        
        # # >>> STORE OUTPUT <<<
        # # convert units of mass balance terms
        # self.runoff = runoff / DENSITY_WATER
        # self.melt = melt / DENSITY_WATER
        # self.refreeze = refreeze / DENSITY_WATER
        # self.accum = snowfall / DENSITY_WATER
        # self.rainfall = rainfall / DENSITY_WATER

        # # store timestep data
        # self.output.store_timestep(self,enbal,surface,layers,time)   

        # # >>> END TIMESTEP <<<
        # # debugging: print current state and monthly melt at the start of each month
        # if time.is_month_start and time.hour == 0 and self.args.debug:
        #     self.current_state_prints()

        # define the next state
        next_state = current_state # ._replace()

        # pack outputs
        out = {
            # 'melt_energy': melt_energy,
            # 'shortwave_in': SWin, 'shortwave_ref': SWout,
            # 'longwave_in': LWin, 'longwave_out': LWout,
            # 'sensible_heat': sensible_heat, 'latent_heat': latent_heat,
            # 'rain_heat': rain_heat, 'ground_heat': ground_heat,

            # 'melt': melt, 'refreeze': refreeze, 'runoff': runoff, 
            # 'accumulation': accumulation, 'rainfall': rainfall,
            # 'sublimation': sublimation, 'deposition': deposition,
            # 'evaporation': evaporation, 'condensation': condensation,
            # 'cumrefreeze': cumrefreeze,

            'airtemp': current_forcings.tempC, 
            'rh': current_forcings.rh,
            'wind': current_forcings.wind,
            'winddir': current_forcings.winddir,
            'tp': current_forcings.tp,
            'sp': current_forcings.sp,
        }

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

        step_records = StepOutputs(**out)
        return next_state, step_records
    
    # execute model with jax.lax.scan
    final_state, records = jax.lax.scan(
        step, initial_state, all_forcings
    )

    # ===== COMPLETED SIMULATION: STORE DATA =====
    return final_state, records