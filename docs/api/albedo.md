# Albedo (`pebsi/albedo.py`)

Computes broadband snow and ice albedo using a neural network emulator of the SNICAR (Snow, Ice, and Aerosol Radiative transfer) model.

---

## `get_albedo(state, params, forcings)`

Main entry point. Assembles the input vector for each point (grain size, LAP concentrations, solar geometry, cloud cover) and runs it through the SNICAR emulator. Returns a broadband albedo value per point `(N_POINTS,)`.

Called once per day at the time(s) specified by `albedo_TOD` in the config.

---

## Internal helpers

- **`_load_emulator(path)`** — loads the pre-trained SNICAR neural network weights from the `.npz` file at startup
- **`_forward(x)`** — runs a single forward pass through the emulator; JIT-compiled via JAX
