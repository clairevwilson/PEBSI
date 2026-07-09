# Mass Balance (`pebsi/massbalance.py`)

`MassBalanceDriver` manages all subsurface and surface mass processes. It is called once per timestep from the main loop.

---

## Top-level drivers

| Method | When called | What it does |
|--------|------------|--------------|
| `run_new_mass(state, forcings)` | Every timestep | Adds accumulation and dry deposition; returns updated state and precipitation flux dict |
| `run_vertical_processes(state, forcings, fluxes)` | Every timestep | Runs the full subsurface pipeline: melting → percolation → particle routing → refreezing → phase changes → layer checking → temperature profile |
| `run_state_updates(state, forcings)` | Every timestep | Updates grain size, densification, and surface roughness |
| `run_daily_routines(state, forcings)` | Once per day | Updates albedo-related daily trackers |
| `run_annual_routines(state, forcings)` | Once per year | Converts old snow to firn; resets annual trackers |

---

## Accumulation

- **`get_precip_amounts(forcings)`** — partitions total precipitation into snowfall and rainfall using a linear temperature threshold
- **`add_accumulation(snowfall, rainfall, state, forcings)`** — inserts a new snow layer at the top of the column

---

## Subsurface processes

- **`heating_melting(state, fluxes)`** — distributes melt energy (surface + penetrating SW) through layers; converts ice to meltwater
- **`percolation(state, fluxes)`** — routes meltwater and rainfall downward through the column; generates runoff when layers reach irreducible water content
- **`route_particles(state, forcings, fluxes)`** — transports BC, OC, and dust with percolating water
- **`refreezing(state)`** — refreezes liquid water in sub-zero layers; releases latent heat
- **`phase_changes(state, latent_heat)`** — handles sublimation, deposition, evaporation, and condensation at the surface
- **`resolve_temperature_profile(state)`** — solves the 1D heat equation through the column (Crank-Nicholson)

---

## Layer evolution

- **`densification(state)`** — compacts snow and firn layers (Boone, Herron-Langway, or Kojima scheme)
- **`evolve_grain_size(state, forcings)`** — dry and wet grain metamorphosis via lookup table
- **`roughness(state)`** — updates surface aerodynamic roughness length based on surface type and time since snowfall
- **`end_of_summer(state)`** — detects the end of the melt season and converts old snow layers to firn

---

## Deposition

- **`add_dry_deposition(state, forcings)`** — adds dry BC, OC, and dust fluxes to the surface layer
