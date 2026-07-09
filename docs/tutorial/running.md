# Running a Simulation

```bash
python simulation.py [options]
```

## Command-line arguments

| Short | Long | Type | Description |
|-------|------|------|-------------|
| `-c` | `--use_config` | bool | Use the configuration file. If `True`, specify the filename with `--config_fn`, otherwise defaults to `config.yaml`. |
| `-cf` | `--config_fn` | str | Path to a configuration file. Setting this automatically enables `--use_config`. |
| `-id` | `--rgi_ids` | list | RGI glacier IDs, e.g. `['01.00570']`. |
| `-start` | `--start_date` | str | Start date for the simulation. |
| `-end` | `--end_date` | str | End date for the simulation. |
| `-pb` | `--progress_bar` | bool | Show a progress bar for the main loop. |
| | `-store_data` | bool | Save the output to disk. |
| | `-debug` | bool | Print debug statements. |
| `-out` | `--output_fn` | str | Filename for the stored output. |
| | `-use_aws` | bool | Use AWS (weather station) forcing data. |

## Examples

Run the built-in test simulation:

```bash
python simulation.py --testing
```

Run a specific glacier with a config file:

```bash
python simulation.py --config_fn my_config.yaml
```

Run a glacier by RGI ID over a date range, storing output:

```bash
python simulation.py -id 01.00570 -start 2015-01-01 -end 2020-12-31 -store_data
```
