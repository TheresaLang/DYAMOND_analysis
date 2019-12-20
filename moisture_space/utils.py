# Useful functions to manipulate atmospheric quantities given in moisture space.
# Unless specified differenty all variable fields should have the dimensions (zlevs, profiles),
# where zlevs are the atmospheric levels and profiles are the atmospheric profiles. 

import numpy as np
import typhon
import processing_tools
from netCDF4 import Dataset
from os.path import join
from scipy.interpolate import interp1d


def get_quantity_at_level(field, height, level):
    """ Returns value(s) at a given altitude level.
    
    Parameters:
        field (2darray): variable field with dimensions (zlevs, x)
        height (1darray): height [m] with dimension (zlevs) 
        level (numeric): target level [m]
    
    Returns:
        1darray, dimensions (x): quantity at target level        
    """
    
    num_profiles = field.shape[1]
    target_value = np.ones(num_profiles) * np.nan
    for p in range(num_profiles):
        target_value[p] = interp1d(height, field[:, p])(level)
    
    return target_value

def get_height_array(height, shape):
    """ Create an ndarray of heights from a 1darray.
    
    Parameters:
        height (1darray): height vector 
        shape (tupel): Shape of output ndarray
        
    Returns:
        2darray, dimensions (zlevs, x): height array
    """
    if len(shape) > 1:
        shape_hor = shape[1:]
        h = np.repeat(height, np.prod(shape_hor))
        h = np.reshape(h, shape)
    elif len(shape) == 1:
        h = height
    
    return h

def calc_vertical_mean(field, height):
    """ Calc a vertical average weighted with layer depth.
    
    Parameters:
        field (2darray): variable field with dimensions (levs, x)
        height (1darray): height field with dimension (levs) 
        
    Returns:
        2darray, dimensions (levs, x): vertical mean of field
    """
    
    layer_depth = np.diff(height)
    field_halflevels = field[:-1] + 0.5 * np.diff(field, axis=0)
    
    weighted_average = np.sum(field_halflevels * np.expand_dims(layer_depth, 1), axis=0)\
    / np.sum(np.expand_dims(layer_depth, 1))
    return weighted_average

def calc_UTH(relative_humidity, specific_humidity, temperature, pressure, height, thres_upper_bnd = 0.018, thres_lower_bnd=1.2):
    """ Calculate upper tropospheric humidity.
    
    Parameters:
        relative_humidity (2darray): relative humidity [-]
        specific_humidity (2darray): specific humidity [kg/kg]
        temperature (2darray): temperature [K]
        pressure (2darray): pressure [Pa]
        height (1darray): height [m]
        thres_upper_bnd (numeric): IWV threshold [kg/m^2] for upper boundary of UTH layer (default: 0.018 kg/m^2)
        thres_lower_bnd (numeric): IWV threshold [kg/m^2] for lower boundary of UTH layer (default: 1.2 kg/m^2)
        
    Returns:
        2darray, dimensions (levs, x): UTH [-]
    """
    array_shape = temperature.shape 

    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        relative_humidity = np.flip(relative_humidity, axis=0)
        specific_humidity = np.flip(specific_humidity, axis=0) 
        temperature = np.flip(temperature, axis=0)
        pressure = np.flip(pressure, axis=0)
        
    iwv_above = np.zeros(specific_humidity.shape)    
    for lev in range(specific_humidity.shape[0]):
        print(lev)
        iwv_above[lev] = calc_IWV(specific_humidity[lev:], temperature[lev:], pressure[lev:], height[lev:])
    
    #print(iwv_above)
    uth_idx = np.logical_and(iwv_above >= thres_upper_bnd, iwv_above <= thres_lower_bnd)
    relative_humidity_for_uth = relative_humidity.copy()
    relative_humidity_for_uth[~uth_idx] = np.nan
    h = get_height_array(height, array_shape)
    dz = np.diff(h, axis=0)
    dz[~uth_idx[1:]] = np.nan
    uth = np.nansum(relative_humidity[:-1] * dz, axis=0) / np.nansum(dz, axis=0)
    
    return uth

def calc_UTH_boundaries(relative_humidity, specific_humidity, temperature, pressure, height, thres_upper_bnd = 0.018, thres_lower_bnd=1.2):
    """ Calculate boundaries of the upper tropospheric humidity (UTH) layer.
    
    Parameters:
        relative_humidity (2darray): relative humidity [-]
        specific_humidity (2darray): specific humidity [kg kg**-1]
        temperature (2darray): temperature [K]
        pressure (2darray): pressure [Pa]
        height (1darray): height [m]
        thres_upper_bnd (numeric): IWV threshold [kg m**-2] for upper boundary of UTH layer (default: 0.018 kg m**-2)
        thres_lower_bnd (numeric): IWV threshold [kg m**-2] for lower boundary of UTH layer (default: 1.2 kg m**-2)
        
    Returns:
        1darray, dimensions (x): lower boundary [m]
        1darray, dimensions (x): upper boundary [m]
    """

    array_shape = temperature.shape 
    num_levels = len(height)
    height_ind = array_shape.index(num_levels)
    if height_ind == 0:
        profile_shape = array_shape[1:]
        if len(array_shape) == 2:
            dim_order = (0, 1)
        elif len(array_shape) == 3:
            dim_order = (0, 1, 2)
    else:
        profile_shape = array_shape[:-1]
        if len(array_shape) == 2:
            dim_order = (1, 0)
        elif len(array_shape) == 3:
            dim_order = (1, 2, 0)
        
    num_profiles = np.prod(profile_shape)
    relative_humidity = np.reshape(relative_humidity.transpose(dim_order), (num_levels, num_profiles))
    specific_humidity = np.reshape(np.transpose(specific_humidity, dim_order), (num_levels, num_profiles))
    temperature = np.reshape(np.transpose(temperature, dim_order), (num_levels, num_profiles))
    pressure = np.reshape(np.transpose(pressure, dim_order), (num_levels, num_profiles))
    threshold_lower_boundary = 1.2
    threshold_upper_boundary = 0.018

    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        relative_humidity = np.flip(relative_humidity, axis=0)
        specific_humidity = np.flip(specific_humidity, axis=0) 
        temperature = np.flip(temperature, axis=0)
        pressure = np.flip(pressure, axis=0)
        
    iwv_above = np.zeros(specific_humidity.shape)    
    for lev in range(specific_humidity.shape[0]):
        iwv_above[lev] = calc_IWV(specific_humidity[lev:], temperature[lev:], pressure[lev:], height[lev:])
    
    uth = np.ones(num_profiles) * np.nan
    lower_boundary = np.ones(num_profiles) * np.nan
    upper_boundary = np.ones(num_profiles) * np.nan
    
    for p in range(num_profiles):
        height_iwv_interp = interp1d(iwv_above[:, p], height, bounds_error=False, fill_value='extrapolate')
        lower_boundary[p] = height_iwv_interp(threshold_lower_boundary)
        upper_boundary[p] = height_iwv_interp(threshold_upper_boundary)
        uth_heights = np.linspace(lower_boundary[p], upper_boundary[p], 20)
        rh_interp = interp1d(height, relative_humidity[:, p], bounds_error=False, fill_value='extrapolate')
        uth[p] = np.mean(rh_interp(uth_heights))
    #uth_idx = np.logical_and(iwv_above >= 0.018, iwv_above <= 1.2)
    
    #print(uth_idx.shape)
    #relative_humidity[~uth_idx] = np.nan
    #h = get_height_array(height, array_shape)
    #dz = np.diff(h, axis=0)
    #dz[~uth_idx[1:]] = np.nan
    #print(dz.shape)
    #uth = np.nansum(relative_humidity[:-1] * dz, axis=0) / np.nansum(dz, axis=0)
    
    uth = np.reshape(uth, (profile_shape))
    lower_boundary = np.reshape(lower_boundary, (profile_shape))
    upper_boundary = np.reshape(upper_boundary, (profile_shape))
    
    return lower_boundary, upper_boundary

def calc_IWV(specific_humidity, temperature, pressure, height):
    """ Calculate integrated water vapor (IWV).
    
    Parameters:
        specific_humidity (2darray): Specific humidity [kg/kg]
        temperature (2darray): Temperature [K]
        pressure (2darray): Pressure [Pa]
        height (1darray): Height vector [m]
    
    Returns: 
        1darray, dimensions (x): integrated water vapor [kg m**-2]
    """
    # get shape of input arrays
    array_shape = temperature.shape    
    # generate height array
    h = get_height_array(height, array_shape)    
    # gas constant for water vapor
    R_v = typhon.constants.gas_constant_water_vapor
    # calculate vmr
    vmr = typhon.physics.specific_humidity2vmr(specific_humidity)
    # calculate water vapor density
    rho = typhon.physics.thermodynamics.density(pressure, temperature, R=R_v)  
    rho[np.where(np.isnan(rho))] = 0.
    vmr[np.where(np.isnan(vmr))] = 0.
    # if surface corresponds to first entry of array, the arrays have to be flipped
    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        vmr = np.flip(vmr, axis=0)
        rho = np.flip(rho, axis=0)    
    # integrate vertically
    iwv = np.trapz(vmr * rho, height, axis=0)
    
    return iwv

def calc_IWP(ice_mass_mixing_ratio, temperature, pressure, specific_humidity, height):
    """ Calculate Ice Water Path (IWP) by integrating the product of ice mass mixing ratio and
    the density of moist air over height.
    
    Parameters:
        ice_mass_mixing_ratio (2darray): Mass mixing ratio of ice [kg/kg]
        temperature (2darray): Temperature [K]
        pressure (2darray): Pressure [Pa]
        specific_humidity (2darray): Specific humidity [kg/kg]
        height (1darray): Height vector [m]
        
    Returns:
        2darray, dimensions (zlevs, x): ice water path [kg m**-2]
    """
    # get shape of input arrays
    array_shape = temperature.shape    
    # generate height array
    h = get_height_array(height, array_shape)    
    # define constants
    R_dry = typhon.constants.gas_constant_dry_air
    R_wv = typhon.constants.gas_constant_water_vapor
    # Gas constant of moist air
    R_moist = R_dry * (1 + (R_wv / R_dry - 1) * specific_humidity)
    # calculate density
    density = pressure / R_moist / temperature   
    density[np.isnan(density)] = 0.
    ice_mass_mixing_ratio[np.isnan(ice_mass_mixing_ratio)] = 0.          
    # if surface corresponds to first entry of array, the arrays have to be flipped
    if height[0] > height[-1]:
        h = np.flip(h, axis=0)
        ice_mass_mixing_ratio = np.flip(ice_mass_mixing_ratio, axis=0)
        density = np.flip(density, axis=0)    
    # integrate vertically
    ice_water_path = np.trapz(ice_mass_mixing_ratio * density, h, axis=0)
    return ice_water_path

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
        temperature (2darray): Temperature [K] with dimensions (levs, x)
        specific_humidity (2darray): Specific humidity [kg/kg] with dimensions (levs, x)
        pressure (2darray): Pressure [Pa] with dimensions (levs, x)
        
    Returns:
        2darray, dimensions (zlevs, x): Relative humidity [-]
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