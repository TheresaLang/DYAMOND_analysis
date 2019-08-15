import numpy as np
import typhon
import processing_tools
from netCDF4 import Dataset
from os.path import join

def read_var(infile, varname):
    """ Reads in one variable from a given netCDF file.
    
    Parameters:
        infile (str): Path to input file
        varname (str): Name of netCDF variable to read in
    
    Returns:
        float or ndarray: Data array with variable
    """
    with Dataset(infile) as ds:
        var = ds.variables[varname][:].filled(np.nan)
    
    return var

def select_lat_lon_box(field, lat, lon, lat_bnds, lon_bnds):
    """ Selects a latitude-longitude box from a 3D field with dimensions (height, latitude, longitude).
    The box boundaries can be specified in lat_bnds and lon_bnds. If identical values are provided for
    upper and lower latitude (longitude) boundaries, only values along one latitude (longitude) that is
    closest to this value are selected.
    
    Parameters:
        field (3D array): Variable field with dimensions (height, latitude, longitude)
        lat (1D array): Latitudes corresponding to field in deg North
        lon (1D array): Longitudes corresponding to field in deg East
        lat_bnds (list): Southern and northern box boundaries in deg North (e.g. [-30, 30])
        lon_bnds (list): Eastern and western box boundaries in deg East
        
    Returns:
        3D array: Selected latitude-longitude box from field
    """
    if lat_bnds[0] == lat_bnds[-1]:
        lat_ind = np.argmin(np.abs(lat - lat_bnds[0]))
        lat_box = np.expand_dims(field[:, lat_ind], axis=1)
    else:
        lat_ind = np.where(np.logical_and(lat >= lat_bnds[0], lat <= lat_bnds[-1]))[0]
        lat_box = field[:, lat_ind]
        
    if lon_bnds[0] == lon_bnds[-1]:
        lon_ind = np.argmin(np.abs(lon - lon_bnds[0]))
        lat_lon_box = np.expand_dims(lat_box[:, :, lon_ind], axis=2)
    else:
        lon_ind = np.where(np.logical_and(lon >= lon_bnds[0], lon <= lon_bnds[-1]))[0]
        lat_lon_box = lat_box[:, :, lon_ind]
    
    return lat_lon_box

def select_height_box(field, height, height_bnds):
    """ Selects height levels from a field with dimensions (height, latitude, longitude). 
    The height boundaries have to be specified in height_bnds. If identical values are provided
    for the lower and upper boundaries, only one height that is closest to this value is selected.
    
    Parameters:
        field (3D array): Variable field with dimensions (height, latitude, longitude)
        height (1D array): Heights corresponding to field in m
        height_bounds (list): Lower and upper boundary of height box in m
        
    Returns:
        3D array: Selected height box from field
    """
    if height_bnds[0] == height_bnds[-1]:
        height_ind = np.argmin(np.abs(height - height_bnds[0]))
    else:
        height_ind = np.where(np.logical_and(height >= height_bnds[0], height <= height_bnds[-1]))[0]
    
    height_box = field[height_ind]
    
    return height_box
        
def spec_hum2rel_hum(specific_humidity, temperature, pressure, phase='mixed'):
    """ Calculate relative humidity from specific humidity. 
    
    The equilibrium water vapour pressure can be calculated with respect to 
    water or ice or the mixed-phase (mixed-phase means that the equilibrium pressure 
    over water is taken for temperatures above the triple point, the value over ice 
    is taken for temperatures below -23 K and for intermediate temperatures the 
    equilibrium pressure is computed as a combination of the values over water and 
    ice according to the IFS documentation (https://www.ecmwf.int/node/18714, 
    Chapter 12, Eq. 12.13)
    
    Paramters:
        temperature (float or ndarray): Temperature [K]
        specific_humidity (float or ndarray): Specific humidity [kg/kg]
        pressure (float or ndarray): Pressure [Pa]
        
    Returns:
        float or ndarray: Relative humidity [-]
    """
    
    vmr = typhon.physics.specific_humidity2vmr(specific_humidity)
    if phase == 'mixed':
        e_eq = typhon.physics.e_eq_mixed_mk(temperature)
    elif phase == 'water':
        e_eq = typhon.physics.e_eq_water_mk(temperature)
    elif phase == 'ice':
        e_eq = typhon.physics.e_eq_ice_mk(temperature)
    
    rh = vmr * pressure / e_eq
    
    return rh

def calc_zonal_mean(field, axis=2):
    """ Calculate a zonal mean of a 3D field with dimensions (height, latitude, longitude).
    If the order of the dimensions is different, the axis along which to average can be 
    specified.
    
    Parameters:
        field (3D array): Field with dimensions (height, latitude, longitude)
        axis (int): Axis along which to average, if order of dimensions is not
            (height, latitude, longitude).
            
    Returns:
        2D array: zonal mean of field
    """
    
    return np.nanmean(field, axis=axis)

def calc_mean_profile(field):
    """ Calculate a mean vertical profile for a variable field with dimensions (height, latitude, longitude).
    
    Parameters:
        field (3D array): Variable field with dimensions (height, latitude, longitude)
        
    Returns:
        1D array: Mean vertical profile 
    """
    zonmean = np.nanmean(field, axis=2)
    mean_profile = np.nanmean(zonmean, axis=1)
    
    return mean_profile
    
    