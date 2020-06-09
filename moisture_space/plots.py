# Useful plotting routines for atmospheric profiles in moisture space

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def moisture_space_contourf(fig, ax, x_vector, y_vector, plot_field, contours, x_lims, y_lims, x_label, y_label, cb_label, cb=True, ticks=None, tick_labels=None, cm_orientation='horizontal', cb_extend='both', cb_ticks=None, cb_format=None, **kwargs):
    
    x, h = np.meshgrid(x_vector, y_vector)
    im = ax.contourf(x, h, plot_field, contours, extend=cb_extend, **kwargs)
    #boundaries=[-10]+list(contours)+[10]
    if cb:
        if cb_format is None:
            cb_format = '%.2f'
        cb = fig.colorbar(im, orientation=cm_orientation, ax=ax, format=cb_format, ticks=cb_ticks) #format='%.5f'
        cb.set_label(cb_label)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if ticks is not None:
        tick_idx = []
        for t in tick_labels:
            tick_idx.append(np.argmin(np.abs(ticks - t)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)
        for label in ax.xaxis.get_ticklabels()[-2:]:
            label.set_visible(False)
    
    return im
    
def moisture_space_contour(fig, ax, x_vector, y_vector, plot_field, contours, x_lims, y_lims, x_label, y_label, color, ticks=None, tick_labels=None, **kwargs):
    x, h = np.meshgrid(x_vector, y_vector)
    c = ax.contour(x, h, plot_field, contours, colors=color)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if ticks is not None:
        tick_idx = []
        for t in tick_labels:
            tick_idx.append(np.argmin(np.abs(ticks - t)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)
        for label in ax.xaxis.get_ticklabels()[-2:]:
            label.set_visible(False)
    
    return c
    
def moisture_space_line(ax, x_vector, y_vector, x_lims, y_lims, x_label, y_label, **kwargs):
    l = ax.plot(x_vector, y_vector, **kwargs)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return l

def percentile_profiles(perc_start, perc_end, step, moisture_space_collection, scaling_factor, x_limits, x_label, y_limits, y_label):
    fig, ax = plt.subplots(1, len(range(perc_start, perc_end+1, step)), figsize=(17, 5), sharey=True)
    for space in moisture_space_collection.moisture_spaces:
        for i, p in enumerate(np.arange(perc_start, perc_end+1, step)):
            ax[i].plot(space.mean_profile(p - step // 2, p + step // 2) * scaling_factor, space.levels * 1e-3, label=space.name)
            ax[i].set_xlim(x_limits)
            ax[i].set_xlabel(x_label)
            ax[i].set_title(f'{p-step//2}-{p+step//2}')
            seaborn.despine(top=True, right=True, ax=ax[i])
    ax[0].set_ylim(y_limits)
    ax[0].set_ylabel(y_label)  
    ax[0].legend()
    
def diff_percentile_profiles(perc_start, perc_end, step, moisture_space_collection, scaling_factor, x_limits, x_label, y_limits, y_label):
    fig, ax = plt.subplots(1, len(range(perc_start, perc_end+1, step))-1, figsize=(17, 5), sharey=True)
    for space in moisture_space_collection.moisture_spaces: 
        for i, p in enumerate(np.arange(perc_start, perc_end, step)):
            ax[i].plot(space.mean_profile(p+step-step//2, p+step+step//2) * scaling_factor - space.mean_profile(p-step//2, p+step//2) * scaling_factor, space.levels * 1e-3, label=space.name)
            ax[i].set_xlim(x_limits)

            ax[i].set_title(f'{p+step} - {p}')
            ax[i].set_xlabel('Diff')
            seaborn.despine(top=True, right=True, left=True, ax=ax[i])
            ax[i].plot(np.zeros(space.num_levels), space.levels * 1e-3, color='k', lw=0.5)
            
    ax[0].set_ylim(y_limits)
    ax[0].set_ylabel(y_label)
    
def DYAMOND_colors(experiment):
    colors = {
        'ICON-2.5km': 'C0',
        'ICON-5.0km_1': 'gold',
        'NICAM-3.5km': 'C1',
        'SAM-4.0km': 'C7',
        'UM-5.0km': 'C6',
        'FV3-3.25km': 'C5',
        'GEOS-3.0km': 'C2',
        'IFS-4.0km': 'C3',
        'IFS-9.0km': 'C9',
        'MPAS-3.75km': 'C4',
        'ARPEGE-2.5km': 'C8',
        'ERA5-31.0km': 'k'
    }
    
    return colors[experiment]

def axis_labels(variable):
    axis_labels = {
        'TEMP': 'Temperature / K',
        'PRES': 'Pressure / hPa',
        'QI': 'Cloud ice content / mg kg$^{-1}$',
        'QI_vol': 'Cloud ice content / mg m$^{-3}$',
        'QC': 'Cloud water content / mg kg$^{-1}$',
        'QV': 'Specific humidity / g kg$^{-1}$',
        'RH': 'Relative humidity / %',
        'W': 'Vertical velocity m s$^{-1}$',
        'OLR': 'OLR / W m$^-2$',
        'STOA': 'Net SW TOA / W m$^-2$',
        'H_tropo': 'Tropopause_height / km',
        'CFI': 'Ice cloud fraction / %',
        'CFL': 'Liquid cloud fraction / %',
        'ICQI': 'In-cloud ice content / mg kg$^{-1}$',
        'LR': 'Lapse rate K km$^{-1}$',
        'logQV': 'ln(q)',
        'S': 'Static stability K hPa$^{-1}$',
        'THETA_E': r'$\theta_e$ / K',
        'THETA_ES': r'$\theta_e*$ / K',
        'IWV': r'IWV / kg m$^{-2}$'
    }
    
    return axis_labels[variable]

def scaling_factors(variable):
    scaling_factors = {
        'TEMP': 1,
        'PRES': 1e-2,
        'QI': 1e6,
        'QI_vol': 1e6,
        'QC': 1e6,
        'QV': 1e3,
        'RH': 1e2,
        'W': 1,
        'OLR': 1,
        'H_tropo': 1e-3,
        'CFI': 1e2,
        'CFL': 1e2,
        'ICQI': 1e6,
        'LR': 1e3,
        'logQV': 1,
        'S': 1e2,
        'THETA_E': 1,
        'THETA_ES': 1,
        'IWV': 1
    }
    return scaling_factors[variable]