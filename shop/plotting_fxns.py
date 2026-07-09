import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mean_squared_error = lambda model, data: np.mean(np.square(model - data))

mpl.style.use('seaborn-v0_8-white')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

# (label, type, units) → list of vars that share those properties
_vargroups = {
    ('Temperature', 'Temperature', 'C'):          ['surftemp', 'airtemp'],
    ('Mass balance', 'MB', 'm w.e.'):             ['melt', 'runoff', 'accumulation', 'refreeze', 'MB'],
    ('Heat fluxes',  'Flux', 'W m$^{-2}$'):       ['melt_energy', 'shortwave_in', 'shortwave_ref',
                                                    'longwave_in', 'longwave_out',
                                                    'longwave_net', 'shortwave_net',
                                                    'sensible_heat', 'latent_heat',
                                                    'rain_heat', 'ground_heat'],
    ('Temperature',  'Layers', 'C'):              ['layertemp'],
    ('Density',      'Layers', 'kg m$^{-3}$'):    ['layerdensity'],
    ('Mass',         'Layers', 'kg m$^{-2}$'):    ['layerice'],
    ('Water content','Layers', 'kg m$^{-2}$'):    ['layerwater'],
    ('Black carbon', 'Layers', 'ppb'):            ['layerBC'],
    ('Organic carbon','Layers','ppb'):            ['layerOC'],
    ('Dust',         'Layers', 'ppm'):            ['layerdust'],
    ('Grain size',   'Layers', 'um'):             ['layergrainsize'],
    ('Layer height', 'Layers', 'm'):              ['layerheight'],
    ('Layer refreeze','Layers','kg m-2'):         ['layerrefreeze'],
    ('Layer age',    'Layers', 'days'):           ['layerage'],
    ('Layer type',   'Layers', '-'):              ['layertype'],
    ('Surface height change', 'MB', 'm$'):        ['dh'],
    ('Albedo',       'Albedo', '-'):              ['albedo'],
}
varprops = {var: {'label': label, 'type': type_, 'units': units}
            for (label, type_, units), vars in _vargroups.items()
            for var in vars}

# colormap and colorbar bounds for each layer variable
_layer_bounds = {
    'layerBC':        [-5,   30],
    'layerOC':        [-5,  100],
    'layerdust':      [0,    50],
    'layerdensity':   [50,  800],
    'layerwater':     [-1,   15],
    'layertemp':      [-10,   0],
    'layergrainsize': [100, 1500],
    'layerrefreeze':  [-1,   20],
    'layerheight':    [0,     2],
    'layertype':      [0,     2],
    'layerage':       [0,   120],
}

_layer_cmaps = {
    'layerBC':        'Greys',
    'layerOC':        'Oranges',
    'layerdust':      'Reds',
    'layertemp':      'plasma',
    'layerdensity':   'Greens',
    'layerwater':     'Blues',
    'layergrainsize': 'PuRd',
    'layerrefreeze':  'Purples',
    'layerheight':    'magma',
    'layertype':      'viridis',
    'layerage':       'OrRd',
}


def simple_plot(ds, time, vars, res='d', t='', cumMB=True,
                skinny=True, save_fig=False, new_y=['None'], date_form=None):
    """
    Returns a simple timeseries plot of the variables as lumped in the input.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset object containing the model output
    vars : list-like
        List of strings where the variables to be plotted together are nested together
        e.g. [['airtemp','surftemp'],['SWnet','LWnet','sensible','latent']]
    time : list-like
        Either len-2 list of start date, end date, or a list of datetimes
    res : str
        Abbreviated time resolution (e.g. '12h' or 'd')
    t : str
        Title for the figure
    skinny : Bool
        True or false, defines the height of each panel
    save_fig : Bool
        False or filepath to save the image
    new_y : list-like
        List of variables in vars that should be plotted on a new y-axis
    """
    h = 2 if skinny else 4
    fig, axes = plt.subplots(len(vars), 1, figsize=(8, h*len(vars)), sharex=True, layout='constrained')

    # accept either a [start, end] shorthand or an explicit time index
    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end   = pd.to_datetime(time[1])
        time  = pd.date_range(start, end, freq='h')
    ds = ds.sel(time=time)
    ds_mean = ds.resample(time=res).mean(dim='time', keep_attrs='units')
    ds_sum  = ds.resample(time=res).sum(dim='time',  keep_attrs='units')

    # single color iterator shared across all panels so each line gets a unique color
    c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
    for i, v in enumerate(vars):
        axis = axes[i] if len(vars) > 1 else axes
        for var in np.array(v):
            try:
                c = next(c_iter)
            except StopIteration:
                c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
                c = next(c_iter)

            # MB vars shown as cumulative sum; layer vars collapsed to surface layer
            if var in ['melt','runoff','accum','refreeze','dh','MB'] and cumMB:
                var_to_plot = ds_sum[var].cumsum()
            elif 'layer' in var:
                var_to_plot = ds_mean[var].isel(layer=0)
            else:
                var_to_plot = ds_mean[var]

            if var in new_y:
                newaxis = axis.twinx()
                newaxis.plot(ds_mean.coords['time'], var_to_plot, color=c, label=var)
                newaxis.grid(False)
                newaxis.set_ylabel({varprops[var]['label']})
                newaxis.legend(bbox_to_anchor=(1.01, 1.1), loc='upper left')
            else:
                axis.plot(ds_mean.coords['time'], var_to_plot, color=c, label=var)
                axis.set_ylabel(varprops[var]['label'])

        axis.tick_params(length=5)
        axis.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    if date_form is None:
        date_form = mpl.dates.DateFormatter('%d %b')
    elif type(date_form) == str:
        date_form = mpl.dates.DateFormatter(date_form)
    axis.xaxis.set_major_formatter(date_form)
    axis.set_xlim(start, end)
    fig.suptitle(t)
    if save_fig:
        plt.savefig(save_fig, dpi=150)
    return fig, axes


def plot_hours(ds, vars, skinny=False, time=False):
    h = 1.5 if skinny else 2
    fig, axes = plt.subplots(len(vars), 1, figsize=(7, h*len(vars)), sharex=True, layout='constrained')
    ds['hour'] = (['time'], pd.to_datetime(ds['time'].values).hour)

    # accept either a [start, end] shorthand or an explicit time index
    if not time:
        start = ds.time.values[0]
        end = ds.time.values[-1]
        time = pd.date_range(start, end, freq='h')
    elif len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start, end, freq='h')
    ds = ds.sel(time=time)

    for i, v in enumerate(vars):
        axis = axes[i] if len(vars) > 1 else axes
        vararray = np.array(v)
        c_iter = iter([all_colors[j] for j in range(8)])
        for var in vararray:
            try:
                c = next(c_iter)
            except StopIteration:
                c_iter = iter([all_colors[j] for j in range(8)])
                c = next(c_iter)

            # average each hour of day across all days in the selected period
            var_hourly = []
            for hour in np.arange(24):
                ds_hour = ds.where(ds['hour'] == hour, drop=True)
                vardata = ds_hour.isel(layer=0)[var].to_numpy() if 'layer' in var else ds_hour[var].to_numpy()
                hourly_mean = np.mean(vardata)
                if 'melt' in vararray or 'MB' in vararray:
                    hourly_mean *= 1000
                var_hourly.append(hourly_mean)

            axis.plot(np.arange(24), var_hourly, label=var, color=c)
            axis.legend(ncols=2 if len(vararray) > 4 else 1)
            axis.set_xlim(0, 23)
            if 'SWnet' in vararray or 'SWin' in vararray:
                axis.set_ylabel('Flux (W m$^{-2}$)')
            elif 'melt' in vararray or 'MB' in vararray:
                axis.set_ylabel('Mass balance (mm w.e.)')
            elif 'surftemp' in vararray or 'airtemp' in vararray:
                axis.set_ylabel('Temperature ($^{\circ}$C)')
    axis.set_xlabel('Hour of Day')


def layer_heatmap(ds, dates, vars, force_layers=False,
                     t='', plot_ax=False,
                     plot_firn=True, plot_ice=False, ylim=False,
                     colorbar=True, diverging=False):
    """
    force_layers:
        Three options:
        - False, takes all snow layers
        - List of integers to select those layer indices
        - Depth in m
    """
    diff = dates[1] - dates[0]   # bar width = timestep spacing

    fig, axes = plt.subplots(len(vars), figsize=(5, 1.7*len(vars)), sharex=True, layout='constrained')
    if plot_ax:
        assert len(plot_ax) == len(vars), f"plot_ax should be length {len(vars)}"
        axes = plot_ax
    if len(vars) == 1 and '__iter__' not in dir(axes):
        axes = [axes]

    # density cutoff that determines which layers are shown
    dens_lim = 1000 if plot_ice else (890 if plot_firn else 600)

    for i, var in enumerate(vars):
        assert 'layer' in var, 'choose layer variable'
        ax     = axes[i]
        bounds = _layer_bounds[var].copy()
        ctype  = _layer_cmaps[var]

        if diverging:
            ctype  = 'coolwarm'
            bounds = (-100, 100) if var == 'layergrainsize' else (-1, 1) if var == 'layertemp' else (-10, 10)

        # tighten density range when firn is excluded
        if var == 'layerdensity' and not plot_firn:
            bounds = [0, 500]

        norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
        cmap = mpl.colormaps[ctype]

        for step in dates:
            height  = ds.sel(time=step)['layerheight'].to_numpy()
            vardata = ds.sel(time=step)[var].to_numpy()
            dens    = ds.sel(time=step)['layerdensity'].to_numpy()

            if isinstance(force_layers, bool):
                layers_to_plot = np.where(dens < dens_lim)[0]
            elif hasattr(force_layers, '__iter__'):
                layers_to_plot = force_layers
            else:
                layers_to_plot = np.where(np.cumsum(height) < force_layers)[0]
            if plot_ice:
                layers_to_plot = np.arange(len(vardata))

            # flip so the deepest layer is at the bottom of the stacked bar
            height  = np.flip(height[layers_to_plot])
            vardata = np.flip(vardata[layers_to_plot])
            dens    = np.flip(dens[layers_to_plot])

            # convert stored mass to more interpretable units
            if var == 'layerwater':
                porosity = 1 - dens / 900
                vardata  = vardata / (porosity * 1000 * height) * 100  # → % saturation
            if var == 'layerrefreeze':
                vardata = vardata / (dens * height) * 100              # → % of layer mass

            bottom = 0
            for dh, data in zip(height, vardata):
                if np.isnan(dh):
                    continue
                color = cmap(norm(data))
                if 'density' in var and data > 899:
                    color = '0.1'   # render ice layers as near-black
                ax.bar(step, dh, bottom=bottom, width=diff, color=color, linewidth=0.5, edgecolor='none')
                bottom += dh

        if colorbar:
            sm  = mpl.cm.ScalarMappable(cmap=ctype, norm=plt.Normalize(bounds[0], bounds[1]))
            leg = plt.colorbar(sm, ax=ax, aspect=7)
            leg.ax.tick_params(labelsize=9)
            if 'BC' in var:
                leg.ax.set_ylim(0, 30)
                leg.ax.set_yticks([0, 15, 30])
            label = varprops[var]['label'] + ' (' + varprops[var]['units'] + ')'
            leg.set_label(label, rotation=270, labelpad=27, fontsize=12)

        ax.grid(axis='y')
        ax.tick_params(length=5)
        if ylim:
            ax.set_ylim(ylim)

    ylabel = 'Height above ice (m)'
    if len(axes) > 1:
        fig.supylabel(ylabel)
    else:
        axes[0].set_ylabel(ylabel)
    fig.suptitle(t, fontsize=14)

    # default to bimonthly ticks; switch to hourly if the window is < 5 days
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %d'))
    ax.set_xticks(pd.date_range(dates[0], dates[-1], freq='2MS'))
    ax.set_xlim([dates[0], dates[-1]])

    if dates[-1] - dates[1] < pd.Timedelta(days=5):
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%m/%d %H:00'))
        ax.set_xticks(pd.date_range(dates[0], dates[-1], 5))

    if not plot_ax:
        return fig, axes
    else:
        plt.close()
        return axes
