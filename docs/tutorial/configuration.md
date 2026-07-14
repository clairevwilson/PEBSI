# Configuration

The `config.yaml` file is the core of every simulation. You can copy it or edit it directly.

## Minimum requirements

The bare minimum to run a new glacier is to set the RGI glacier ID:

```yaml
rgi_ids: ["01.00570"]  # example: Gulkana Glacier, Alaska
```

This is also the place to specify your local filepaths (e.g., the path to your climate data or DEM), physics options (e.g., use penetrating shortwave radiation?), and parameters (e.g., precipitation gradient or temperature lapse rate).

## Parameter hierarchy

Settings are resolved in this order:

1. `util/defaults.py`
2. `config.yaml`
3. Command-line arguments

Any parameter listed in `defaults.py` can be added to `config.yaml`. More information on the parameter and physics options can be found in `defaults.py`.

## Command-line overrides

The most commonly adjusted parameters can be passed directly on the command line, which is useful for scripted runs over many glaciers or date ranges. See [Running a Simulation](running.md) for the full list of arguments.

## Specifying simulation points

To set up the spatial grid of a simulation, there are currently two options.

### 1. `method_distribute='sites'`

This option is for simulating named, individual sites, such as mass balance index sites or the location of an automatic weather station. 

To access sites by name, add new rows to the `data/glacier_metadata.csv` file. At minimum, specify the RGI ID, site name, and the point coordinates in decimal degrees. To run those sites, specify `sites` argument in the configuration file.

The `glacier_metadata.csv` file can also store elevation (m.a.s.l.) and slope and aspect (degrees) if you have in situ measurements that are more accurate than what the DEM can provide.

### 2. `method_distribute='grid'`

This option is for a distributed, gridded run across a glacier or region where the user specifies the number of points to simulate. The grid is uniform in space; i.e., the number of points per glacier scales with glacier's area.

Options within the grid method:

- `n_points`: approximate number of points to distribute across all glaciers in the simulation
- `min_area`: for running an entire region, this sets the minimum area of glaciers to include

## Spatially-variant parameters 

(Only applies for `method_distribute='sites'`) Certain parameters can vary by site if the config file contains a list of the length `N_SITES`. This option is to enable usage of calibrated parameters. Currently supported are the following variables:

- Climate variable downscaling: `kp`, `wind_factor`, `dust_factor`, `lapse_rate`, `precgrad`
- Local albedo: `albedo_ice`, `albedo_fresh_snow`
- Surface roughness lengths: `roughness_aging_rate`, `roughness_fresh_snow`, `roughness_firn`, `roughness_ice`
- Particle partitioning: `ksp_BC`, `ksp_OC`, `ksp_dust` 
- Initial snow and firn depth: `initial_snow_depth`, `initial_firn_depth`