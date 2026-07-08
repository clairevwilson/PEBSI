import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray  # noqa: F401  (activates .rio accessor)

dep_types = ['bcwet', 'bcdry', 'ocwet', 'ocdry']

MERRA_varnames = {
    'bcwet': 'BCWT002', 'ocwet': 'OCWT002',
    'bcdry': 'BCDP002', 'ocdry': 'OCDP002',
}

UKESM_varnames = {
    'bcwet': 'tendency_of_atmosphere_mass_content_of_elemental_carbon_dry_aerosol_particles_due_to_wet_deposition',
    'bcdry': 'tendency_of_atmosphere_mass_content_of_elemental_carbon_dry_aerosol_particles_due_to_dry_deposition',
    'ocwet': 'tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_particles_due_to_wet_deposition',
    'ocdry': 'tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_particles_due_to_dry_deposition',
}

MERRA_scale_maps = {
    'bcwet': 'reg01_BC_regression_map.nc',
    'bcdry': 'reg01_BC_regression_map.nc',
    'ocwet': 'reg01_OC_regression_map.nc',
    'ocdry': 'reg01_OC_regression_map.nc',
}

UKESM_particle = {'bcwet': 'bc', 'bcdry': 'bc', 'ocwet': 'oc', 'ocdry': 'oc'}
UKESM_deptype  = {'bcwet': 'wet', 'bcdry': 'dry', 'ocwet': 'wet', 'ocdry': 'dry'}

fp_base  = '/Users/cvw/local/climate_data/'
fp_merra = os.path.join(fp_base, 'MERRA2/')
fp_ukesm = os.path.join(fp_base, 'UKESM/dr401_GFED/')
fp_out   = fp_merra

region = 'reg01'

# Collect per-var results for the debug plots
debug = {}

for var in dep_types:
    print(f'\n=== {var} ===')
    merra_vn = MERRA_varnames[var]
    ukesm_vn = UKESM_varnames[var]
    particle  = UKESM_particle[var]
    deptype   = UKESM_deptype[var]

    # ------------------------------------------------------------------ #
    # 1. Load MERRA-2 and apply existing scale map (BC002 → BC001+BC002) #
    # ------------------------------------------------------------------ #
    fn_merra = os.path.join(fp_merra, region, f'{merra_vn}_{region}.zarr')
    ds_merra = xr.open_zarr(fn_merra, consolidated=False, chunks={})

    fn_map = os.path.join(fp_merra, MERRA_scale_maps[var])
    ds_map = xr.open_dataset(fn_map)
    ratio_map = ds_map['ratio']

    if 'wet' in var:
        ratio_map = 1

    da_merra = ds_merra[merra_vn] * ratio_map

    # ------------------------------------------------------------------ #
    # 2. Load UKESM                                                        #
    # ------------------------------------------------------------------ #
    fn_ukesm = os.path.join(fp_ukesm, f'sum_{particle}_{deptype}deposition_kgm-2s-1.nc')
    ds_ukesm = xr.open_dataset(fn_ukesm)
    da_ukesm = ds_ukesm[ukesm_vn]

    if da_ukesm['longitude'].values.max() > 180:
        da_ukesm = da_ukesm.assign_coords(
            longitude=((da_ukesm['longitude'] + 180) % 360) - 180
        ).sortby('longitude')

    # ------------------------------------------------------------------ #
    # 3. Find overlapping years                                            #
    # ------------------------------------------------------------------ #
    merra_years = set(da_merra.time.dt.year.values)
    ukesm_years = set(da_ukesm.time.dt.year.values)
    overlap_years = sorted(merra_years & ukesm_years)
    assert len(overlap_years) > 0, 'No overlapping years between MERRA-2 and UKESM'
    print(f'  Overlapping years: {overlap_years[0]}–{overlap_years[-1]} ({len(overlap_years)} years)')

    # ------------------------------------------------------------------ #
    # 4. Compute per-year annual sums (rate × dt → kg m-2 yr-1)          #
    # ------------------------------------------------------------------ #
    MERRA_DT = 3600.0
    UKESM_DT = 86400.0

    def yearly_sums(da, dt, years):
        da_years = da.sel(time=da.time.dt.year.isin(years))
        return (da_years * dt).resample(time='YS').sum(skipna=False)

    print('  Computing MERRA-2 annual sums ...')
    merra_yearly = yearly_sums(da_merra, MERRA_DT, overlap_years).compute()  # (year, lat, lon)

    print('  Computing UKESM annual sums ...')
    ukesm_yearly = yearly_sums(da_ukesm, UKESM_DT, overlap_years).compute()  # (year, latitude, longitude)

    # ------------------------------------------------------------------ #
    # 5. Regrid UKESM → MERRA-2 grid (nearest neighbour, per year)       #
    # ------------------------------------------------------------------ #
    merra_lats = merra_yearly['lat'].values
    merra_lons = merra_yearly['lon'].values

    ukesm_regridded = ukesm_yearly.sel(
        latitude=xr.DataArray(merra_lats, dims='lat'),
        longitude=xr.DataArray(merra_lons, dims='lon'),
        method='nearest'
    ).assign_coords(lat=merra_lats, lon=merra_lons)  # (year, lat, lon)

    # ------------------------------------------------------------------ #
    # 6. Scale factor from mean annual sums                               #
    # ------------------------------------------------------------------ #
    merra_annual   = merra_yearly.mean('time')
    ukesm_annual   = ukesm_regridded.mean('time')

    scale_factor = merra_annual / ukesm_annual

    eps = 1e-30
    scale_factor = scale_factor.where(
        (np.abs(ukesm_annual) > eps) & (np.abs(merra_annual) > eps)
    )

    # ------------------------------------------------------------------ #
    # 7. R² per grid cell across years                                    #
    # ------------------------------------------------------------------ #
    m = merra_yearly.values          # (n_years, lat, lon)
    u = ukesm_regridded.values       # (n_years, lat, lon)

    m_mean = m.mean(axis=0, keepdims=True)
    u_mean = u.mean(axis=0, keepdims=True)
    m_c = m - m_mean
    u_c = u - u_mean

    num   = (m_c * u_c).sum(axis=0)
    denom = np.sqrt((m_c**2).sum(axis=0) * (u_c**2).sum(axis=0))

    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = np.where(denom > 0, (num / denom) ** 2, np.nan)

    r2_da = xr.DataArray(r2, coords={'lat': merra_lats, 'lon': merra_lons}, dims=['lat', 'lon'])

    # ------------------------------------------------------------------ #
    # 8. Save                                                              #
    # ------------------------------------------------------------------ #
    fn_out = os.path.join(fp_out, f'ukesm_merra2_{region}_{particle}{deptype}.nc')
    scale_factor.name = 'ratio'
    scale_factor.attrs = {
        'long_name': f'UKESM-to-MERRA-2 scale factor for {var}',
        'description': (
            'Multiplicative factor applied to UKESM deposition to remove mean annual bias '
            'relative to MERRA-2. MERRA-2 was pre-scaled using the existing BC/OC regression map.'
        ),
        'merra2_variable': merra_vn,
        'ukesm_variable': ukesm_vn,
        'overlap_years': f'{overlap_years[0]}-{overlap_years[-1]}',
        'units': '1',
    }
    scale_factor.to_netcdf(fn_out)
    print(f'  Saved → {fn_out}')

    debug[var] = {
        'scale_factor': scale_factor,
        'r2': r2_da,
        'lats': merra_lats,
        'lons': merra_lons,
    }

    ds_merra.close()
    ds_map.close()
    ds_ukesm.close()

# ------------------------------------------------------------------ #
# Load RGI 01 glaciers and clip data to touching cells               #
# ------------------------------------------------------------------ #
fp_rgi01 = '/Users/cvw/local/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
rgi01 = gpd.read_file(fp_rgi01).to_crs('EPSG:4326')

# Use first var's grid (all vars share the same MERRA-2 lat/lon)
_lats = debug[dep_types[0]]['lats']
_lons = debug[dep_types[0]]['lons']

def clip_to_rgi(da):
    """Clip a (lat, lon) DataArray to RGI region 01, keeping all touching cells."""
    da = da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    da = da.rio.write_crs('EPSG:4326')
    return da.rio.clip(rgi01.geometry, all_touched=True, drop=False)

proj = ccrs.PlateCarree()
states = cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces', '10m',
    edgecolor='black', facecolor='none', linewidth=0.8
)

def make_map_axes(fig, row, col, idx):
    ax = fig.add_subplot(row, col, idx, projection=proj)
    ax.set_facecolor('#888888')
    ax.set_extent([_lons.min(), _lons.max(), _lats.min(), _lats.max()], crs=proj)
    ax.add_feature(states)   # drawn first, behind the heatmap
    return ax

def add_panel(ax, data, **mesh_kwargs):
    """Plot clipped heatmap on top of the state outlines."""
    clipped = clip_to_rgi(data)
    im = ax.pcolormesh(_lons, _lats, clipped.values,
                       shading='auto', transform=proj, **mesh_kwargs)
    return im

# ------------------------------------------------------------------ #
# Debug plot 1: scale factor heatmap (one panel per var)             #
# ------------------------------------------------------------------ #
fig1 = plt.figure(figsize=(13, 8))
fig1.suptitle('UKESM→MERRA-2 scale factor', fontsize=13)
norm_sf = plt.matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)

for i, var in enumerate(dep_types, 1):
    ax = make_map_axes(fig1, 2, 2, i)
    im = add_panel(ax, debug[var]['scale_factor'], norm=norm_sf, cmap='RdBu')
    fig1.colorbar(im, ax=ax, label='scale factor', shrink=0.75)
    ax.set_title(var)

fig1.tight_layout()
fig1.savefig(os.path.join(fp_out, 'debug_scale_factor_map.png'), dpi=150)
print('\nSaved debug_scale_factor_map.png')

# ------------------------------------------------------------------ #
# Debug plot 2: R² map (one panel per var)                           #
# ------------------------------------------------------------------ #
fig2 = plt.figure(figsize=(13, 8))
fig2.suptitle('UKESM vs MERRA-2 inter-annual R² (per grid cell)', fontsize=13)

for i, var in enumerate(dep_types, 1):
    ax = make_map_axes(fig2, 2, 2, i)
    im = add_panel(ax, debug[var]['r2'], vmin=0, vmax=1, cmap='RdYlGn')
    fig2.colorbar(im, ax=ax, label='R²', shrink=0.75)
    ax.set_title(var)

fig2.tight_layout()
fig2.savefig(os.path.join(fp_out, 'debug_r2_map.png'), dpi=150)
print('Saved debug_r2_map.png')

plt.close()
print('\nDone.')
