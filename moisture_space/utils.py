# Useful functions to manipulate atmospheric quantities given in moisture space.
# Unless specified differenty all variable fields should have the dimensions (zlevs, profiles),
# where zlevs are the atmospheric levels and profiles are the atmospheric profiles. 

import numpy as np
import typhon
import processing_tools
from netCDF4 import Dataset
from os.path import join
from scipy.interpolate import interp1d
from scipy.signal import detrend

def eofs(series_array):
    """ Calculate Empirical Orthogonal Functions (EOFs) as well as
    expansion coefficients and the fraction of overall variability 
    explained by each EOF.
    
    Parameters:
        series_array (2darray): array with first dimension corresponding to
            continuous variable (e.g. time, models)
        num (int): number of expansion coefficients to calculate 
            (default: 5)
   
    """
    # detrend
    array_detrended = detrend(series_array, axis=0, type='constant').T
    # covariance matrix
    cov_matrix = np.cov(array_detrended)
    # Eigenvalues and -vectors
    eigvals, EOFs = np.linalg.eig(cov_matrix)
    # reshape
    EOF_frac = eigvals / np.sum(eigvals)   
    # calculate expansion coefficients corresponding to first N EOFs

    expansion_coeff = np.ones((EOFs.shape[0], series_array.shape[0])) * np.nan
    for i in range(EOFs.shape[0]):
        expansion_coeff[i] = np.matmul(array_detrended.T, EOFs[:, i])
        
    return EOFs, EOF_frac, expansion_coeff

def svd(series_arrays):
    """ Perform a singular value decomposition.
    """
    arrays_detrended = [((series_arrays[i] - np.mean(series_arrays[i], axis=0)) ) for i in range(2)]#/ np.std(series_arrays[i], axis=0)
    cov_matrix = np.matmul(arrays_detrended[0].T, arrays_detrended[1])
    
    u, s, vh = np.linalg.svd(cov_matrix)

    expansion_coeff_1 = np.matmul(arrays_detrended[0], u)
    expansion_coeff_2 = np.matmul(arrays_detrended[1], vh.T)
    
    return u, s, vh, expansion_coeff_1, expansion_coeff_2

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

def get_mean_quantity_in_layer(field, height, layer_bnds):
    """ Returns mean value of a quantity over a given altitude range.
    
    Parameters:
        field (2darray): variable field with dimensions (zlevs, x)
        height (1darray): height [m] with dimension (zlevs) 
        layer_bnds (list of num): target level [m]
    
    Returns:
        1darray, dimensions (x): mean quantity in layer     
    """
    height_ind = np.logical_and(height >= layer_bnds[0], height <= layer_bnds[1])
    mean_quantity = np.mean(field[height_ind, :], axis=0)
    
    return mean_quantity

def get_quantity_at_half_levels(field):
    """ Returns field at height levels marking the middle between 
    former height levels.
    
    Parameters:
        field (2darray): variable field with dimensions (zlevs, x)

    Returns:
        2darray, dimensions (zhalflevs, x): field at half levels
    """
    field_at_half_levels = field[:-1] + 0.5 * np.diff(field)
    
    return field_at_half_levels

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
        2darray, dimensions (x): vertical mean of field
    """
    
    layer_depth = np.diff(height)
    field_halflevels = field[:-1] + 0.5 * np.diff(field, axis=0)
    
    weighted_average = np.sum(field_halflevels * np.expand_dims(layer_depth, 1), axis=0)\
    / np.sum(np.expand_dims(layer_depth, 1))
    return weighted_average

def calc_IWV_above(specific_humidity, temperature, pressure, height):
    """ Calculate IWV above each level.
    
    Parameters:
        specific_humidity (2darray): specific humidity [kg/kg]
        temperature (2darray): temperature [K]
        pressure (2darray): pressure [Pa]
        height (1darray): height [m]
        
    Returns:
        2darray, dimensions (levs, x): UTH [-]
    """
    array_shape = temperature.shape 

    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        specific_humidity = np.flip(specific_humidity, axis=0) 
        temperature = np.flip(temperature, axis=0)
        pressure = np.flip(pressure, axis=0)
        
    iwv_above = np.zeros(specific_humidity.shape)    
    for lev in range(specific_humidity.shape[0]):
        iwv_above[lev] = calc_IWV(specific_humidity[lev:], temperature[lev:], pressure[lev:], height[lev:])
    
    return iwv_above
    

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

def calc_IWV(specific_humidity, temperature, pressure, height, height_start=0, height_end=20000):
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
    # heights to integrate over
    height_ind = np.logical_and(height >= height_start, height <= height_end)
    # integrate vertically
    iwv = np.trapz(vmr[height_ind] * rho[height_ind], height[height_ind], axis=0)
    
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

def calc_dewpoint(e):
    """ Calculate the dew point temperature from the water vapor pressure e.
    
    Parameters:
        e (2darray): Water vapor pressure [Pa]
        
    Returns:
        2darray: Dew point temperature [K]
    """
    
    t_d = 243.5 * np.log(e / 611.2) / (17.67 - np.log(e / 611.2)) + 273.15
    
    return t_d

def drh_dt(temperature, pressure, relative_humidity, vertical_velocity):
    """
    """
    L = typhon.constants.heat_of_vaporization
    R = typhon.constants.gas_constant_water_vapor
    g = typhon.constants.g
    cp = typhon.constants.isobaric_mass_heat_capacity
    density = pressure / R / temperature
    Gamma = -g / cp
    drh_dt = (relative_humidity / pressure * -density * g - relative_humidity * L / R / temperature ** 2 * Gamma) * vertical_velocity
    
    return drh_dt
    
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

def rel_hum2spec_hum(relative_humidity, temperature, pressure, phase='mixed'):
    """ Calculate the specific humidity from a given relative humidity.
    """
    if phase == 'mixed':
        e_eq = typhon.physics.e_eq_mixed_mk(temperature)
    elif phase == 'water':
        e_eq = typhon.physics.e_eq_water_mk(temperature)
    elif phase == 'ice':
        e_eq = typhon.physics.e_eq_ice_mk(temperature)
        
    vmr = relative_humidity * e_eq / pressure
    specific_humidity = typhon.physics.vmr2specific_humidity(vmr)
    
    return specific_humidity
    

def interpolate_vertically(field, height, target_height):
    """ Interpolate a filed to a new height vector. 
        
    Parameters:
        field (2darray): variable field with dimensions (levs, x)
        height (1darray): height field with dimension (levs)
        target_height (1darray): new height field with dimensions (levs_new)
        
    Returns:
        2darray, dimensions (levs_new, x): vertical mean of field
    """
    
    num_profiles = field.shape[1]
    field_interp = np.ones((target_height.shape[0], field.shape[1])) * np.nan
    
    for p in range(num_profiles):
        field_interp[:, p] = interp1d(height, field[:, p], fill_value='extrapolate', bounds_error=False)(target_height)
    
    return field_interp

def tropopause_height(temperature, height, min_height=6e3, max_height=20e3):
    """ Calculate tropopause height according to WMO definition: 
    The lowest level at which the lapse rate decreases to 2 C/km or less, 
    provided that the average lapse rate between this level and all higher 
    levels within 2 km does not exceed 2 C/km."
    
    Parameters:
        temperature (2darray): Temperature [K] with dimensions (levs, x)
        height (1darray): Height [m] with dimension (levs)
        min_height (numeric): Lowest possible tropopause height [m], default: 6000
        max_height (numeric): Highest possible tropopause height [m], default: 20000
        
    Returns:
        1darray, dimension (x): Tropopause height [m]
    """
    if len(temperature.shape) > 1:
        num_profiles = temperature.shape[1]
    else:
        num_profiles = 1
    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        temperature = np.flip(temperature, axis=0)
    
    # height at half levels
    height_half = get_quantity_at_half_levels(height)
    # indices of heights between min_height and max_height
    ind_pos_height = np.logical_and(height_half > min_height, height_half < max_height)
    
    tropo_height = np.ones(num_profiles) * np.nan    
    for p in range(num_profiles):
        # lapse rate
        lapse_rate = - np.diff(temperature[:, p]) / np.diff(height)
        # possible tropopause heights (heights where lapse rate < 2K/km)
        pos_tropo_ind = np.where(lapse_rate[ind_pos_height] <= 2e-3)[0]
        # Check whether the mean lapse rate in a 2km-layer above is also < 2K/km
        # If not, go to next possible tropopause height
        for k in range(len(pos_tropo_ind)):
            th = height_half[ind_pos_height][pos_tropo_ind[k]]
            ind_layer_above = np.where(np.logical_and(height_half >= th, height_half <= th + 2e3))
            mean_lr_above = calc_vertical_mean(np.expand_dims(lapse_rate[ind_layer_above], 1), height_half[ind_layer_above])
            if mean_lr_above <= 2e-3:
                tropo_height[p] = th
                break
               
    return tropo_height

def cold_point_tropopause(temperature, height, min_height=6e3, max_height=20e3):
    """ Calculate the cold point tropopause height and tempertaure.
    """
    ind_pos_height = np.logical_and(height > min_height, height < max_height)
    
    ind_tropo = np.argmin(temperature[:, ind_pos_height], axis=1)
    tropo_temp = temperature[:, ind_pos_height][:, ind_tropo]
    tropo_height = height[ind_pos_height][ind_tropo]
    
    return tropo_height, tropo_temp

def rh_peak_height(relative_humidity, height, min_height=6e3, max_height=20e3):
    """ Returns the height at which the relative humidity is at its maximum (between min_height
    and max_height).
    
    Parameters:
        relative_humidity (2darray): Relative humidity [-] with dimensions (levs, x)
        height (1darray): Height [m] with dimension (levs)
        min_height (numeric): Lowest possible tropopause height [m], default: 6000
        max_height (numeric): Highest possible tropopause height [m], default: 20000
    
    Returns:
        1darray, dimension (x): Height of maximum RH [m]
    """
    num_profiles = relative_humidity.shape[1]
    if height[0] > height[-1]:
        height = np.flip(height, axis=0)
        relative_humidity = np.flip(relative_humidity, axis=0)
        
    rh_peak_height = np.ones(num_profiles) * np.nan 
    rh_at_peak = np.ones(num_profiles) * np.nan 
    # indices of heights between min_height and max_height
    ind_pos_height = np.logical_and(height > min_height, height < max_height)
    height = height[ind_pos_height]
    relative_humidity = relative_humidity[ind_pos_height]
    
    for p in range(num_profiles):
        rh_peak_height[p] = height[np.argmax(relative_humidity[:, p])]
        rh_at_peak[p] = np.max(relative_humidity[:, p])
    
    return rh_peak_height, rh_at_peak
    
def calc_density_moist_air(pressure, temperature, specific_humidity):
    """ Calculate the density of moist air.
   
    Parameters:
       pressure: pressure in Pa
       temperature: temperature in K
       specific_humidity: specific_humidity in kg/kg
    """
    num_profiles = pressure.shape[0]
    density = np.ones(pressure.shape) * np.nan
    for i in range(num_profiles):        
        mixing_ratio = typhon.physics.specific_humidity2mixing_ratio(specific_humidity[i]) 
        # virtual temperature
        virt_temp = temperature[i] * (1 + 0.61 * mixing_ratio) 
        # air density (from virtual temperature)
        density[i] = typhon.physics.density(pressure[i], virt_temp)
    
    return density


def calc_integrated_velocity(vertical_velocity, height, h_start, h_end, temperature, pressure, specific_humidity):
    """ Calculate vertically integrated mass weighted velocity.
    """
    num_profiles = vertical_velocity.shape[0]
    h_ind = np.logical_and(height > h_start, height < h_end)
    dh = np.gradient(height)
    density = calc_density_moist_air(pressure, temperature, specific_humidity)
    
    integrated_velocity = np.ones(vertical_velocity.shape[0])
    for i in range(num_profiles):
        integrated_velocity[i] = np.trapz(vertical_velocity[i][h_ind] * density[i][h_ind], x=height[h_ind]) / (height[h_ind][-1] - height[h_ind][0])
    
    return integrated_velocity

def calc_pot_temp(temperature, pressure, p0=100000):
    """ Calculate potential temperature.
    """
    Rd = typhon.constants.gas_constant_dry_air
    cpd = typhon.constants.isobaric_mass_heat_capacity
    k = Rd / cpd
    
    pot_temp = temperature * (p0 / pressure) ** k
    
    return pot_temp
    
def calc_equivalent_pot_temp(temperature, pressure, specific_humidity):
    """ Calculate equivalent potential temperature.
    """
    Rd = typhon.constants.gas_constant_dry_air
    cpd = typhon.constants.isobaric_mass_heat_capacity
    Rv = typhon.constants.gas_constant_water_vapor
    Lv = typhon.constants.heat_of_vaporization

    mixing_ratio = typhon.physics.specific_humidity2mixing_ratio(specific_humidity)
    relative_humidity = spec_hum2rel_hum(specific_humidity, temperature, pressure, phase='mixed')
    pot_temp = calc_pot_temp(temperature, pressure)
    
    equivalent_pot_temp = pot_temp * relative_humidity ** (-mixing_ratio * Rv / cpd) * np.exp(Lv * mixing_ratio / cpd / temperature)
    
    return equivalent_pot_temp

def calc_sat_equivalent_pot_temp(temperature, pressure):
    """ Calc saturation value of equivalent potential temperature.
    """
    
    Rd = typhon.constants.gas_constant_dry_air
    cpd = typhon.constants.isobaric_mass_heat_capacity
    Lv = typhon.constants.heat_of_vaporization
    
    e_eq = typhon.physics.e_eq_mixed_mk(temperature)
    vmr = e_eq / pressure
    sat_mixing_ratio = typhon.physics.vmr2mixing_ratio(vmr)
    pot_temp = calc_pot_temp(temperature, pressure)
    
    sat_equivalent_pot_temp = pot_temp * np.exp(Lv * sat_mixing_ratio / cpd / temperature)
   
    return sat_equivalent_pot_temp