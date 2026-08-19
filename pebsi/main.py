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
from pebsi.physics.layers import *
from pebsi.state import make_step_outputs_class, OUTPUT_GROUPS

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
    
    # execute the model (need to use checkpoint for forward/backward solving)
    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, records = jax.lax.scan(
        scan_step, initial_state, all_forcings, unroll=1
    )

    # ===== COMPLETED SIMULATION: STORE DATA =====
    return final_state, records


# Names correspond to the STEP 1-6 blocks in main.step() above (STEP 7 is
# pure mass-conservation bookkeeping on already-computed fluxes, not an
# independent physics call, so it's not a separate probe point).
STAGE_NAMES = {
    1: 'run_new_mass',
    2: 'run_daily_routines',
    3: 'solve_energy_balance',
    4: 'run_vertical_processes',
    5: 'run_state_updates',
    6: 'run_annual_routines',
}


@functools.partial(jax.jit, static_argnames=['static_args', 'stop_after_stage'])
def main_stage_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    stop_after_stage):
    """
    Debug-only variant of `main` for isolating *which* per-timestep physics
    stage (see STAGE_NAMES) introduces a non-finite gradient, as opposed to
    *when* in the simulation it happens -- every stage runs every timestep,
    so a NaN-gradient chunk-bisection over time can't distinguish them.

    Mirrors main()'s step() exactly but returns early after stop_after_stage
    (a static Python int, so untaken stages are never traced -- not just
    masked out). stop_after_stage is 1-6 per STAGE_NAMES. Returns only the
    final state (no StepOutputs), since callers reduce it to a scalar
    themselves for differentiation.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        if stop_after_stage == 1:
            return current_state, None

        current_state = mb.run_daily_routines(current_state, current_forcings)
        if stop_after_stage == 2:
            return current_state, None

        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)
        if stop_after_stage == 3:
            return current_state, None

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
        if stop_after_stage == 4:
            return current_state, None

        current_state, water_squeezed_out = mb.run_state_updates(current_state, current_forcings)
        if stop_after_stage == 5:
            return current_state, None

        current_state = mb.run_annual_routines(current_state, current_forcings)
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


# Sub-calls inside MassBalanceDriver.run_vertical_processes (stage 4 above),
# in call order -- see that method in pebsi/physics/massbalance.py.
VERTICAL_SUBSTAGE_NAMES = {
    1: 'heating_melting',
    2: 'percolation+route_particles',
    3: 'refreezing',
    4: 'phase_changes',
    5: 'check_layer_sizes',
    6: 'resolve_temperature_profile',
}


@functools.partial(jax.jit, static_argnames=['static_args', 'stop_after_substage'])
def main_vertical_substage_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    stop_after_substage):
    """
    Debug-only variant of `main` that runs stages 1-3 exactly as main_stage_probe
    does (unchanged), then truncates *inside* stage 4 (run_vertical_processes)
    after stop_after_substage (see VERTICAL_SUBSTAGE_NAMES), instead of
    treating that whole stage as one opaque call. stop_after_substage is a
    static Python int, 1-6, so untaken sub-calls are never traced.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        # ----- inlined run_vertical_processes, with sub-call truncation -----
        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        if stop_after_substage == 1:
            return current_state, None

        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        if stop_after_substage == 2:
            return current_state, None

        current_state = mb.refreezing(current_state)
        if stop_after_substage == 3:
            return current_state, None

        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )
        if stop_after_substage == 4:
            return current_state, None

        current_state, dead_mass = check_layer_sizes(current_state, params)
        if stop_after_substage == 5:
            return current_state, None

        current_state = mb.resolve_temperature_profile(current_state)
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


# Phases inside check_layer_sizes (substage 5 above) -- see that function in
# pebsi/physics/layers.py.
LAYER_PHASE_NAMES = {
    1: 'dead_layer_zeroing',
    2: 'merge_scan (merge_existing_layers)',
    3: 'split_scan (split_layer)',
}


@functools.partial(jax.jit, static_argnames=[
    'static_args', 'stop_after_phase', 'disable_any_merge', 'restrict_to_site'
])
def main_layer_phase_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    stop_after_phase,
    disable_any_merge=False,
    restrict_to_site=None):
    """
    Debug-only variant of `main` that runs stages 1-3 and vertical-process
    sub-calls 1-4 (heating_melting through phase_changes) exactly as the
    other probes do, then truncates *inside* check_layer_sizes after
    stop_after_phase (see LAYER_PHASE_NAMES). stop_after_phase is a static
    Python int, 1-3. disable_any_merge (static bool) forces off the
    is_thin_any ice-ice merge path, leaving only snow_merge. restrict_to_site
    (static int or None) forces merge_mask False everywhere except that one
    site -- see check_layer_sizes_probe in pebsi/physics/layers.py.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_probe(
            current_state, params, stop_after_phase, disable_any_merge, restrict_to_site
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


# Phases inside merge_existing_layers (called by check_layer_sizes' merge
# scan, phase 2 above) -- see merge_existing_layers_probe in
# pebsi/physics/layers.py for exact boundaries.
MERGE_PHASE_NAMES = {
    1: 'weighted_avg+extensive_sum',
    2: 'shift float-typed vars',
    3: 'shift int-typed vars (ltype, lage)',
    4: 'lheight_recompute_1',
    5: 'add_bottom_layer',
    6: 'lheight_recompute_2',
    7: 'update_layer_props',
}


@functools.partial(jax.jit, static_argnames=['static_args'])
def main_no_merge_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args):
    """
    Debug-only variant of `main`, identical in every stage EXCEPT
    check_layer_sizes is replaced with check_layer_sizes_no_merge_probe
    (dead-layer zeroing + split scan, no merge scan at all). This is the
    real forward/backward simulation, full fidelity, minus exactly one
    thing -- the direct test of whether merge_existing_layers is actually
    what makes the gradient non-finite, as opposed to an artifact of the
    deep chain of truncating/isolating probes used to localize it (see
    jax_optimize.py's single-site real test).

    Not used by any production path.
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_no_merge_probe(current_state, params)
        current_state = mb.resolve_temperature_profile(current_state)

        current_state, water_squeezed_out = mb.run_state_updates(current_state, current_forcings)
        current_state = mb.run_annual_routines(current_state, current_forcings)
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


@functools.partial(jax.jit, static_argnames=['static_args', 'stop_after_merge_phase', 'restrict_to_site'])
def main_merge_phase_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    stop_after_merge_phase,
    restrict_to_site=None):
    """
    Debug-only variant of `main` that runs stages 1-3 and vertical-process
    sub-calls 1-4 exactly as main_layer_phase_probe does, then within the
    merge scan (check_layer_sizes' phase 2) truncates *inside*
    merge_existing_layers after stop_after_merge_phase (see
    MERGE_PHASE_NAMES). No split scan -- this isolates merge_existing_layers'
    own internal structure. stop_after_merge_phase is a static Python int, 1-6.
    restrict_to_site (static int or None) forces the merge to only occur at
    that one site (see check_layer_sizes_merge_internal_probe), for testing
    a single truncated phase on a single clean, isolated merge event.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_internal_probe(
            current_state, params, stop_after_merge_phase, restrict_to_site
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


@functools.partial(jax.jit, static_argnames=['static_args', 'skip_var', 'restrict_to_site'])
def main_merge_phase1_skipvar_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    skip_var,
    restrict_to_site=None):
    """
    Debug-only variant of `main` that runs ONLY merge_existing_layers'
    phase 1 (weighted-average/extensive-sum -- no shift, no downstream at
    all) on a single clean, isolated merge event, skipping the write-back
    of exactly one variable (see merge_existing_layers_phase1_skipvar_probe).
    Isolates which of the 12 phase-1 variables is responsible, since phase
    1 as a whole was confirmed non-finite on a real, ordinary-looking merge.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_phase1_skipvar_probe(
            current_state, params, skip_var, restrict_to_site
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


MERGE_SKIPBLOCK_NAMES = [
    'shift', 'shift_zero_only', 'lheight_recompute_1', 'add_bottom_layer',
    'lheight_recompute_2', 'update_layer_props',
]


@functools.partial(jax.jit, static_argnames=['static_args', 'skip_block', 'restrict_to_site'])
def main_merge_skipblock_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    skip_block,
    restrict_to_site=None):
    """
    Debug-only variant of `main` that runs the merge scan with every field
    fully faithful to production EXCEPT skip_block is a no-op (see
    merge_existing_layers_skipblock_probe) -- unlike main_merge_phase_probe
    (which truncates and compounds torn-state inconsistency across the
    scan), this isolates one block cleanly every merge, every iteration.
    skip_block is a static Python string (one of MERGE_SKIPBLOCK_NAMES) or
    None (baseline, shift everything -- should match production exactly).
    restrict_to_site (static int or None) forces the merge to only occur
    at that one site, for testing a single isolated merge event.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_skipblock_probe(
            current_state, params, skip_block, restrict_to_site
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


# Ordered float-typed subset of all_layer_vars (see
# merge_existing_layers_var_probe in pebsi/physics/layers.py) --
# ltemp, ldensity, lgrainsize, lice, lwater, lBC, lOC, ldust, ldrefreeze,
# lrefreeze, lheight, ldepth, in that order.
MERGE_VAR_NAMES = [
    'ltemp', 'ldensity', 'lgrainsize',
    'lice', 'lwater', 'lBC', 'lOC', 'ldust', 'ldrefreeze', 'lrefreeze',
    'lheight', 'ldepth',
]


@functools.partial(jax.jit, static_argnames=['static_args', 'n_vars_shifted'])
def main_merge_var_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    n_vars_shifted):
    """
    Debug-only variant of `main`, identical to main_merge_phase_probe through
    the weighted-average/extensive-sum step, but shifts only the first
    n_vars_shifted variables of MERGE_VAR_NAMES (in order) instead of all of
    them -- isolates which specific variable's shift introduces a non-finite
    gradient. n_vars_shifted is a static Python int, 0-12.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_var_probe(
            current_state, params, n_vars_shifted
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


@functools.partial(jax.jit, static_argnames=['static_args', 'skip_var'])
def main_merge_skipvar_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    skip_var):
    """
    Debug-only variant of `main` that runs the merge scan with every field
    fully faithful to production EXCEPT skip_var's shift is a no-op (see
    merge_existing_layers_skip_var_probe) -- unlike main_merge_var_probe
    (which truncates a prefix and compounds torn-state inconsistency across
    the scan), this isolates one variable cleanly. skip_var is a static
    Python string (one of all_layer_vars) or None (baseline, shift
    everything -- should match main_merge_phase_probe's phase-7 behavior).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_skipvar_probe(
            current_state, params, skip_var
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state


@functools.partial(jax.jit, static_argnames=['static_args', 'n_vars_shifted'])
def main_merge_nvars_probe(
    initial_state,
    all_forcings,
    point_attrs,
    static_args,
    dynamic_args,
    n_vars_shifted):
    """
    Debug-only variant of `main`, like main_merge_skipvar_probe (full
    downstream fidelity, no early return) but shifts only the first
    n_vars_shifted variables (MERGE_VAR_NAMES order) instead of skipping
    just one -- finds the minimum number of simultaneously shifted
    variables needed to trigger the non-finite gradient, since omitting
    any single one of 12 was insufficient. n_vars_shifted is a static
    Python int, 0-12.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic (PEBSI_STAGE_BISECT family).
    """
    params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(params)
    eb = EnergyBalanceDriver(params)

    def step(current_state, current_forcings):
        current_forcings = domain_expansion(current_forcings, point_attrs, params)

        rainfall, snowfall, current_state = mb.run_new_mass(current_state, current_forcings)
        current_state = mb.run_daily_routines(current_state, current_forcings)
        current_state, fluxes = eb.solve_energy_balance(current_state, current_forcings, point_attrs)

        vert_fluxes = {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'latent_heat': fluxes['latent_heat'],
            'melt_energy': fluxes['melt_energy'],
            'SWnet_penetrating': fluxes['SWnet_penetrating']
        }

        current_state, melt_array, mass_to_route = mb.heating_melting(current_state, vert_fluxes)
        for var, data in mass_to_route.items():
            vert_fluxes[var] = jnp.sum(data, axis=1)
        current_state, melt_runoff, vert_fluxes = mb.percolation(current_state, vert_fluxes)
        current_state = mb.route_particles(current_state, current_forcings, vert_fluxes)
        current_state = mb.refreezing(current_state)
        current_state, condensation_runoff, mass_fluxes = mb.phase_changes(
            current_state, vert_fluxes['latent_heat']
        )

        current_state = check_layer_sizes_merge_nvars_probe(
            current_state, params, n_vars_shifted
        )
        return current_state, None

    scan_step = jax.checkpoint(step) if static_args.differentiable else step
    final_state, _ = jax.lax.scan(scan_step, initial_state, all_forcings, unroll=1)
    return final_state