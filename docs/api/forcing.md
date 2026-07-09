# Forcing (`pebsi/forcing.py`)

Downscales MERRA-2 (or AWS) climate data from the GCM grid cell to each glacier point. All functions operate on a `ClimateState` and return a corrected `ClimateState`.

These are applied inside `domain_expansion()` at the start of each timestep, before the energy and mass balance calculations.

---

## `domain_expansion(forcings, point_attrs, params)`

Top-level entry point. Calls all of the adjustment functions below in sequence and returns the fully downscaled `ClimateState` ready for the energy balance.

---

## Adjustment functions

| Function | What it does |
|----------|-------------|
| `expand_forcings(forcings, point_attrs)` | Broadcasts per-cell MERRA-2 fields to per-point arrays using `cell_idx` |
| `adjust_temperature(forcings, point_attrs, params)` | Lapse-rate correction from GCM cell elevation to point elevation |
| `adjust_precipitation(forcings, point_attrs, params)` | Applies precipitation gradient with elevation and the `kp` scaling factor |
| `adjust_pressure(forcings, point_attrs, params)` | Hypsometric correction from GCM cell to point elevation |
| `adjust_longwave(forcings, point_attrs, params)` | Corrects downwelling LW for the elevation difference between GCM cell and point |
| `apply_parameters(forcings, params)` | Applies wind factor, dust factor, and temperature perturbation scalings |
