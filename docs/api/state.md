# State (`pebsi/state.py`)

Defines the four immutable JAX-compatible `NamedTuple` classes that carry all data through the model's main loop.

---

## `GlacierState`

A snapshot of the physical state of every point and layer. Updated at each timestep by the mass balance driver.

- **Per-point** `(N_POINTS,)`: surface temperature, albedo, roughness, last snowfall index, and several annual/cumulative trackers (`annual_min_albedo`, `annual_max_snow`, `days_since_snowfall`, `cum_mass_error`, `basal_reservoir`, `past_snow`)
- **Per-layer** `(N_POINTS, N_LAYERS)`: height, depth, type masks, density, temperature, age, grain size, ice/water/refreeze content, and light-absorbing particle (LAP) concentrations (BC, OC, dust)

---

## `ClimateState`

One timestep of climate forcing for every point. Produced by `pack_forcings()` in `simulation.py` and passed read-only into the main loop.

- **Scalar time fields**: `time_idx`, `year`, `month`, `day`, `hour`, `doy`
- **Per-point** `(N_POINTS,)`: `temp`, `tp`, `wind`, `sp`, `rh`, `tcc`, `local_hour`
- **Radiation**: `shortwave_in`, `longwave_in`, `shadow_mask`, `solar_azimuth`, `solar_zenith`
- **Deposition**: dry and wet fluxes for BC, OC, and dust

---

## `PointAttributes`

Time-invariant spatial metadata. Computed once before the main loop and passed as a static argument to JAX.

- **Per-point** `(N_POINTS,)`: `latitude`, `longitude`, `elevation`, `slope`, `aspect`, `sky_view_factor`, `median_elev`, `cell_idx`
- **Per MERRA-2 cell** `(N_UNIQUE,)`: `gcm_elev`, `temp_elev`, `sp_elev`, `LWin_elev` — reference elevations used for lapse-rate and pressure corrections

---

## `StepOutputs`

The full set of energy balance fluxes, mass balance terms, climate variables, and layer properties written to disk for each timestep.
