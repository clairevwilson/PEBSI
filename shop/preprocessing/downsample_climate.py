"""
Takes in MERRA-2 climate data, downsamples it to a
lower temporal resolution, and re-upsamples it
to hourly, which is saved as a new dataset with flag
_3hr or _daily depending on the downsample resolution.
This tests the effect of forcing PEBSI with CMIP6-like
temporal resolution instead of raw hourly data.

For 3-hourly data, the downsampling is done by taking
the mean across 3-hour intervals (aligned to UTC), and
upsampling is done using linear interpolation between
interval centers. Precipitation and deposition are held
at a constant rate within each interval, and shortwave
is distributed by top-of-atmosphere irradiance.

For daily data, the downsampling is done by taking the
daily mean (local days), plus daily minimum and maximum air temperature,
and upsampling is done based on the variable.
- Air temperature: half-cosine curves joining the daily
  minimum (06:00-07:00) and maximum (15:00-16:00)
- Wind speed, precipitation, deposition: constant rate per hour
- Surface pressure, relative and specific humidity, cloud cover: constant per hour
- Shortwave radiation: daily total distributed by top-of-atmosphere irradiance
- Longwave radiation: daily mean scaled by hourly T^4

At both resolutions, wind keeps the mean vector direction
rescaled to the mean speed. Relative humidity is upsampled
directly rather than recomputed from specific humidity,
since a constant daily specific humidity under the diurnal
temperature cycle supersaturates the early morning hours.
"""
import os
import numpy as np
import xarray as xr

base_fp = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/'
input_fp = base_fp + 'reg01/{v}_reg01.zarr'
output_fp = base_fp + 'reg01_{res}/{v}_reg01_{res}.zarr'

# hours per downsampled block
resolutions = {'3hr': 3, 'daily': 24}

# daily blocks are local (AKST) days; 3-hour blocks stay aligned to UTC
UTC_OFFSET = 9
T_MIN_HOUR = 6
T_MAX_HOUR = 15

# interpolated at 3-hourly, constant at daily
state_vars = ['PS', 'RH2M', 'QV2M', 'CLDTOT']
# constant rate at both resolutions
rate_vars = ['PRECTOTCORR', 'BCDP002', 'BCWT002', 'OCDP002', 'OCWT002', 'DUDP003', 'DUWT003']

# latitudes loaded at once, matching the zarr chunking
BAND = 20


def to_blocks(x, n, lead):
    """
    Pads the time (last) axis with NaN so it starts at a block boundary
    and holds whole blocks, then reshapes to (..., n_blocks, n).
    """
    n_blocks = -(-(lead + x.shape[-1]) // n)
    pad = [(0, 0)] * (x.ndim - 1) + [(lead, n_blocks * n - lead - x.shape[-1])]
    return np.pad(x, pad, constant_values=np.nan).reshape(*x.shape[:-1], n_blocks, n)


def unblock(xb, lead, nt):
    """Inverse of to_blocks."""
    return xb.reshape(*xb.shape[:-2], -1)[..., lead:lead + nt]


def interp_anchors(pos, vals, n_hours, cosine=False):
    """
    Interpolates vals (..., n_anchors) located at hour indices pos to
    every hour index, linearly or with half-cosine segments.
    """
    hours = np.arange(n_hours)
    k = np.clip(np.searchsorted(pos, hours) - 1, 0, len(pos) - 2)
    f = np.clip((hours - pos[k]) / (pos[k + 1] - pos[k]), 0, 1)
    if cosine:
        f = (1 - np.cos(np.pi * f)) / 2
    return vals[..., k] * (1 - f) + vals[..., k + 1] * f


def expand_constant(mean, n, lead, nt):
    """Holds each block mean constant over its hours."""
    return np.repeat(mean, n, axis=-1)[..., lead:lead + nt]


def expand_linear(mean, n, lead, nt):
    """Linearly interpolates between block means placed at the block centers."""
    pos = np.arange(mean.shape[-1]) * n + (n - 1) / 2
    return interp_anchors(pos, mean, mean.shape[-1] * n)[..., lead:lead + nt]


def upsample(x, n, lead, expand):
    return expand(np.nanmean(to_blocks(x, n, lead), axis=-1), n, lead, x.shape[-1])


def temp_daily(x, lead):
    """Half-cosine curves through the daily minimum and maximum temperature."""
    xb = to_blocks(x, 24, lead)
    n_days = xb.shape[-2]
    pos = (np.arange(n_days)[:, None] * 24 + np.array([T_MIN_HOUR, T_MAX_HOUR])).ravel()
    vals = np.stack([np.nanmin(xb, axis=-1), np.nanmax(xb, axis=-1)], axis=-1)
    vals = vals.reshape(*xb.shape[:-2], -1)
    return interp_anchors(pos, vals, n_days * 24, cosine=True)[..., lead:lead + x.shape[-1]]


def longwave_daily(x, temp, lead):
    """Daily mean longwave scaled by hourly T^4 relative to its daily mean."""
    mean = np.nanmean(to_blocks(x, 24, lead), axis=-1, keepdims=True)
    t4 = to_blocks(temp.astype(np.float64) ** 4, 24, lead)
    return unblock(mean * t4 / np.nanmean(t4, axis=-1, keepdims=True), lead, x.shape[-1])


def sun_weights(times, lat, lon):
    """Cosine of the solar zenith angle at each timestamp, floored at zero."""
    doy = times.dayofyear.values
    hour = times.hour.values + times.minute.values / 60
    decl = np.radians(-23.44) * np.cos(2 * np.pi * (doy + 10) / 365)
    hour_angle = np.radians(15 * (hour + lon[..., None] / 15 - 12))
    lat = np.radians(lat[..., None])
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.maximum(cosz, 0)


def shortwave(x, lat, lon, n, lead, times):
    """Distributes each block total across its hours by top-of-atmosphere irradiance."""
    xb = to_blocks(x, n, lead)
    wb = np.nan_to_num(to_blocks(sun_weights(times, lat, lon), n, lead))
    total = np.nansum(xb, axis=-1, keepdims=True)
    w_sum = wb.sum(axis=-1, keepdims=True)
    # blocks with the sun down at every hour midpoint fall back to a constant rate
    out = np.where(w_sum > 0, total * wb / np.where(w_sum > 0, w_sum, 1),
                   np.nanmean(xb, axis=-1, keepdims=True))
    return unblock(out, lead, x.shape[-1])


def wind_component(a, b, n, lead, expand):
    """Block-mean component a of the wind vector (a, b), rescaled so the vector has the block-mean speed."""
    ab, bb = to_blocks(a, n, lead), to_blocks(b, n, lead)
    speed = np.nanmean(np.hypot(ab, bb), axis=-1)
    a_mean, b_mean = np.nanmean(ab, axis=-1), np.nanmean(bb, axis=-1)
    vec = np.hypot(a_mean, b_mean)
    # calm blocks (MERRA-2 has stretches of exactly zero wind) stay calm
    a_out = np.where(vec > 0, a_mean * speed / np.where(vec > 0, vec, 1), 0)
    return expand(a_out, n, lead, a.shape[-1])


def along_time(func, *das, **kwargs):
    """Applies func, which acts on the last (time) axis, to loaded DataArrays."""
    out = func(*(np.moveaxis(da.values, 0, -1) if 'time' in da.dims else da.values for da in das), **kwargs)
    return das[0].copy(data=np.moveaxis(out, -1, 0).astype(np.float32))


def open_var(fp, var):
    return xr.open_zarr(fp, consolidated=False, chunks=None)[var]


def process(var, res, func, *das, **kwargs):
    """Runs func on bands of latitude and appends each band to the output store."""
    fp = output_fp.format(v=var, res=res)
    print(f'Processing {var} to {fp}...')
    for start in range(0, das[0].sizes['lat'], BAND):
        band = [da.isel(lat=slice(start, start + BAND)).load() for da in das]
        out = along_time(func, *band, **kwargs).rename(var)
        out.encoding = {}
        if start == 0:
            out.to_dataset().to_zarr(fp, mode='w', zarr_format=2, consolidated=False,
                                     encoding={var: {'chunks': (8760, BAND, 16)}})
        else:
            out.to_dataset().to_zarr(fp, append_dim='lat', consolidated=False)


for res, n in resolutions.items():
    print(f'\n=== {res} ===')
    os.makedirs(os.path.dirname(output_fp.format(v='T2M', res=res)), exist_ok=True)
    expand = expand_linear if res == '3hr' else expand_constant

    temp = open_var(input_fp.format(v='T2M'), 'T2M')
    times = temp.indexes['time']
    lead = (times[0].hour - UTC_OFFSET) % n

    if res == 'daily':
        process('T2M', res, temp_daily, temp, lead=lead)
    else:
        process('T2M', res, upsample, temp, n=n, lead=lead, expand=expand)

    lw = open_var(input_fp.format(v='LWGAB'), 'LWGAB')
    if res == 'daily':
        lw, temp_up = xr.align(lw, open_var(output_fp.format(v='T2M', res=res), 'T2M'), join='inner')
        process('LWGAB', res, longwave_daily, lw, temp_up, lead=lead)
    else:
        process('LWGAB', res, upsample, lw, n=n, lead=lead, expand=expand)

    sw = open_var(input_fp.format(v='SWGDN'), 'SWGDN')
    lat, lon = xr.broadcast(sw.lat, sw.lon)
    process('SWGDN', res, shortwave, sw, lat, lon, n=n, lead=lead, times=times)

    for var in state_vars + rate_vars:
        method = expand if var in state_vars else expand_constant
        process(var, res, upsample, open_var(input_fp.format(v=var), var), n=n, lead=lead, expand=method)

    u = open_var(input_fp.format(v='U2M'), 'U2M')
    v = open_var(input_fp.format(v='V2M'), 'V2M')
    process('U2M', res, wind_component, u, v, n=n, lead=lead, expand=expand)
    process('V2M', res, wind_component, v, u, n=n, lead=lead, expand=expand)

print('Done.')
