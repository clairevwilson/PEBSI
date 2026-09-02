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
import functools
# External libraries
import jax
import jax.numpy as jnp
# Local libraries
from pebsi.physics.energybalance import EnergyBalanceDriver
from pebsi.physics.massbalance import MassBalanceDriver
from pebsi.forcing import domain_expansion
from pebsi.state import make_step_outputs_class, OUTPUT_GROUPS, AGG_METHOD

@functools.partial(jax.jit, static_argnames=['static_args'])
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

    # build output class from store_vars
    StepOutputs = make_step_outputs_class(params.store_vars)
    requested_fields = set(StepOutputs._fields)

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

        # ===================== STEP 4 =====================
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
        mass_fluxes['mass_balance'] = (
            mass_fluxes['accumulation'] + mass_fluxes['refreeze'] - mass_fluxes['melt']
            + mass_fluxes['deposition'] + mass_fluxes['condensation']
            - mass_fluxes['sublimation'] - mass_fluxes['evaporation']
        )

        # ===================== STEP 6 =====================
        #          annual checks and tracker updates
        current_state = mb.run_annual_routines(current_state, current_forcings)

        # ===================== STEP 7 =====================
        #            mass conservation tracking
        next_mass = fetch_current_mass(current_state)
        mass_fluxes['error'] = mass_conservation(
            current_mass, next_mass, mass_fluxes
        )
        mass_fluxes['total_mass'] = next_mass

        # ===================== OUTPUTS =====================
        next_state = current_state
        out = {}

        # climate fields
        climate_map = {
            'airtemp': current_forcings.temp,
            'rh': current_forcings.rh,
            'wind': current_forcings.wind,
            'winddir': current_forcings.winddir,
            'tp': current_forcings.tp,
            'sp': current_forcings.sp,
            'albedo': current_state.albedo,
        }
        for field, value in climate_map.items():
            if field in requested_fields:
                out[field] = value

        # energy balance fields
        for field, value in fluxes.items():
            if field in requested_fields:
                out[field] = value

        # mass balance fields (convert to m w.e.)
        for field, value in mass_fluxes.items():
            if field in requested_fields:
                out[field] = value / params.density_water

        # surface type
        if 'surftype' in requested_fields:
            out['surftype'] = getattr(next_state, 'surftype')

        # total column liquid water [kg m-2]
        if 'total_water' in requested_fields:
            out['total_water'] = jnp.sum(next_state.lwater, axis=1)

        # layer fields
        for field in OUTPUT_GROUPS['layers']:
            if field in requested_fields:
                state_field = field.replace('layer', 'l')
                values = getattr(next_state, state_field)
                if field in ['layerBC', 'layerOC', 'layerdust']:
                    lheight = next_state.lheight
                    safe_height = jnp.where(lheight > 0, lheight, 1e-6)
                    concentration = values / safe_height
                    values = concentration * (1e6 if field in ['layerBC', 'layerOC'] else 1e3)
                out[field] = values

        step_records = StepOutputs(**out)
        return next_state, step_records

    def aggregate_period(hourly_records):
        """
        Collapses one output period's hourly step records (leading axis =
        hours in the period) down to a single record, using each field's
        AGG_METHOD ('sum', 'mean', 'min', or 'last').
        """
        agg = {}
        for field in hourly_records._fields:
            vals = getattr(hourly_records, field)
            method = AGG_METHOD.get(field, 'last')
            if method == 'sum':
                agg[field] = jnp.sum(vals, axis=0)
            elif method == 'mean':
                agg[field] = jnp.mean(vals, axis=0)
            elif method == 'min':
                agg[field] = jnp.min(vals, axis=0)
            else:
                agg[field] = vals[-1]
        return StepOutputs(**agg)

    # execute the model (need to use checkpoint for forward/backward solving)
    scan_step = jax.checkpoint(step) if static_args.differentiable else step

    steps_per_output = params.steps_per_output
    month_lengths = params.month_lengths

    if month_lengths:
        # monthly: unrolled Python loop over statically-known calendar-month
        # segments, so each month's length can vary within the same chunk
        state = initial_state
        period_records = []
        offset = 0
        for length in month_lengths:
            month_forcings = jax.tree.map(
                lambda x: jax.lax.slice_in_dim(x, offset, offset + length, axis=0),
                all_forcings
            )
            state, hourly_records = jax.lax.scan(scan_step, state, month_forcings, unroll=1)
            period_records.append(aggregate_period(hourly_records))
            offset += length
        final_state = state
        records = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *period_records)

    elif steps_per_output <= 1:
        # native hourly resolution: one output record per scanned step
        final_state, records = jax.lax.scan(
            scan_step, initial_state, all_forcings, unroll=1
        )

    else:
        # daily: nest an inner hourly scan inside an outer scan over fixed-size
        # output periods, so only one record per period is retained on-device
        def period_step(period_state, period_forcings):
            next_state, hourly_records = jax.lax.scan(
                scan_step, period_state, period_forcings, unroll=1
            )
            return next_state, aggregate_period(hourly_records)

        periodized_forcings = jax.tree.map(
            lambda x: x.reshape((x.shape[0] // steps_per_output, steps_per_output) + x.shape[1:]),
            all_forcings
        )
        final_state, records = jax.lax.scan(
            period_step, initial_state, periodized_forcings, unroll=1
        )

    # ===== COMPLETED SIMULATION: STORE DATA =====
    return final_state, records
