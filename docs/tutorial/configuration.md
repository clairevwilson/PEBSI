# Configuration

The `config.yaml` file is the core of every simulation. You can copy it or edit it directly.

## Minimum required

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
