# Useful plotting routines for atmospheric profiles in moisture space


import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def moisture_space_contourf(fig, ax, x_vector, y_vector, plot_field, contours, x_lims, y_lims, x_label, y_label, cb_label, **kwargs):
    
    x, h = np.meshgrid(x_vector, y_vector)
    im = ax.contourf(x, h, plot_field, contours, **kwargs)
    cb = fig.colorbar(im, orientation='horizontal', ax=ax)
    cb.set_label(cb_label)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return im
    
def moisture_space_contour(fig, ax, x_vector, y_vector, plot_field, contours, x_lims, y_lims, x_label, y_label, color):
    x, h = np.meshgrid(x_vector, y_vector)
    c = ax.contour(x, h, plot_field, contours, colors=color)
    
    return c
    
def moisture_space_line(ax, x_vector, y_vector, x_lims, y_lims, x_label, y_label, **kwargs):
    l = ax.plot(x_vector, y_vector, **kwargs)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return l