# Running a Simulation

```bash
python simulation.py [options]
```

## Command-line arguments

| Short | Long | Type | Description |
|-------|------|------|-------------|
| `-c` | `--use_config` | bool | Use the configuration file. If `True`, defaults to `config.yaml`. |
| `-cf` | `--config_fn` | str | Path to a configuration file. Setting this automatically enables `--use_config`. |
| `-id` | `--rgi_ids` | list | RGI glacier IDs, e.g. `['01.00570']`. |
| `-start` | `--start_date` | str | Start date for the simulation. |
| `-end` | `--end_date` | str | End date for the simulation. |
| `-pb` | `--progress_bar` | bool | Show a progress bar for the main loop. |
| | `-store_data` | bool | Save the output to disk. |
| | `-debug` | bool | Print debug statements. |
| `-out` | `--output_fn` | str | Filename for the stored output. |
| | `-use_aws` | bool | Use AWS (weather station) forcing data. |

## Simulation set-up

To set up the spatial grid of a simulation, there are currently two options.

### 1. `method_distribute='sites'`

This option is for simulating named, individual sites, such as mass balance index sites or the location of an automatic weather station. To access those sites by name, add new rows to the `data/glacier_metadata.csv` file. At minimum, specify the RGI ID, site name, and the point coordinates in decimal degrees. To run those sites, specify `sites` argument in the configuration file.

The `glacier_metadata.csv` file can also store elevation (m.a.s.l.) and slope and aspect (degrees) if you have in situ measurements that are more accurate than what the DEM can provide.

Also consider storing pre-processed parameters to columns in this file, such as local ice albedo. While PEBSI will not directly access those parameters, a simple simulation processing script can load arrays of individual site parameters into the `config.yaml` file.

### 2. `method_distribute='scatter'`

This option is for a distributed, gridded run across a glacier or region. The grid is uniform in space; i.e., the number of points per glacier is scaled to each glacier's area.

Options within the scatter method:

- `n_points`: approximate number of points to distribute across all glaciers in the simulation
- `min_area`: for running an entire region, this sets the minimum area of glaciers to include

## Examples

Run the built-in test simulation:

```bash
python simulation.py --testing
```

Run a specific glacier or region with a config file:

```bash
python simulation.py --config_fn my_config.yaml
```

Run a glacier by RGI ID over a date range, storing output:

```bash
python simulation.py -id 01.00570 -start 2015-01-01 -end 2020-12-31 -store_data
```
