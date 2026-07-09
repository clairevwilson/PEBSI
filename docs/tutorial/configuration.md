# Configuration

The `config.yaml` file is the core of every simulation. You can copy it or edit it directly.

## Minimum required

The bare minimum to run a new glacier is to set the RGI glacier ID:

```yaml
rgi_id: "01.00570"  # example: Gulkana Glacier, Alaska
```

## Parameter hierarchy

Settings are resolved in this order (later overrides earlier):

1. `util/defaults.py` — model defaults
2. `config.yaml` — your configuration file
3. Command-line arguments — highest priority

Any parameter listed in `defaults.py` can be added to `config.yaml`.

## Command-line overrides

The most commonly adjusted parameters can be passed directly on the command line, which is useful for scripted runs over many glaciers or date ranges. See [Running a Simulation](running.md) for the full list of arguments.
