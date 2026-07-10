# Digital Elevation Model (DEM)

PEBSI uses a DEM to compute terrain shading, which affects the shortwave radiation reaching the glacier surface. The DEM must cover not just the glacier outline but also the surrounding terrain so that ridge shading is computed correctly.

## Using an existing DEM

If you already have a DEM that covers both the glacier and its surrounding ridges, point to it in `config.yaml`:

```yaml
dem_fn: /path/to/your/dem.tif
```

## Downloading a DEM

If you don't have a DEM, use the provided script to download a Copernicus DEM tile for your region:

```bash
python shading/get_cop.py
```
