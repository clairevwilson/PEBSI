# Energy Balance (`pebsi/energybalance.py`)

`EnergyBalanceDriver` computes the surface energy balance at each timestep and returns the surface temperature and net energy available for melt.

!!! note
    These methods are described in further detail in the [technical reference](https://docs.google.com/document/d/1skLi2KsmpXVVr0Mw3aYVu41SMf5OGHMGAQyU9AGE_iM/edit?tab=t.0#heading=h.yflmwwzexoft).

---

## `solve_energy_balance(state, forcings, point_attrs)`

Top-level entry point. Calls `compute_fluxes` iteratively (Brent's method) to find the surface temperature at which the energy balance closes. Returns the updated `GlacierState` (with new `surftemp`) and a dictionary of individual flux terms.

## `compute_fluxes(surftemp_guess, state, forcings, point_attrs)`

Evaluates all flux components at a given surface temperature. Returns net energy and the individual terms needed for the solver and for output.

---

## Individual flux methods

| Method | Returns |
|--------|---------|
| `get_SW(state, forcings, point_attrs)` | Incoming and net shortwave [W m⁻²], accounting for slope/aspect, terrain shading, diffuse fraction, and albedo |
| `get_LW(state, forcings)` | Incoming and outgoing longwave [W m⁻²] via Stefan-Boltzmann |
| `get_turbulent(state, forcings, point_attrs)` | Sensible and latent heat fluxes [W m⁻²] via Bulk Richardson |
| `get_rain(state, forcings)` | Heat flux from rainfall [W m⁻²] |
| `get_ground(state)` | Ground heat flux [W m⁻²] |

---

## Utility methods

- **`sat_vapor_pressure(airtemp)`** — saturation vapour pressure [Pa] using the August-Roche-Magnus formula
- **`diffuse_fraction(rad_glob, solar_zenith, doy)`** — partitions global SW into direct and diffuse components (Wohlfahrt method)
