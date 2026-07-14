# Output data

PEBSI stores output data in the .zarr data format with one file per site / grid cell in the simulation.

## Storage options 

The variables stored within the file can be specified within `config.yaml` using options from the list:
- MB: mass balance terms (melt, accumulation, refreeze, rainfall, phase change fluxes, surface height change)
- EB: energy balance fluxes (shortwave_in/ref, longwave_in/out, sensible and latent heat, rain heat, ground heat)
- climate: subset of climate forcings (airtemp, wind, rh, sp, tp)
- layers: vertical layer states (layertemp, layerdensity, layergrainsize, layerwater, layerBC/OC/dust, layerrefreeze)

The layer data, if included, explodes file size since the model stores 50 layers at a default per point. This is useful for troubleshooting results, but is recommended to be excluded from large simulations to avoid slow write times and huge consumption of disk space.

## Working with outputs

A simple notebook `visualize_output.ipynb` is available which provides a couple of example functions that are useful to inspect the output. Functions call from `shop/plotting_fxns` which includes:

- `simple_plot(ds, vars, time)`: a simple timeseries plot. Variables can be called within groups to plot within the same axis.
- `plot_hours(ds, vars, time)`: a plot which averages each var per hour of day. Variables can be called within groups to plot within the same axis.
- `layer_heatmap(ds, dates, vars)`: a heatmap of the layer distribution as stacked bar charts to visualize how the snowpack evolved.
