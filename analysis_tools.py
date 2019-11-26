import numpy as np
import typhon
import processing_tools
from netCDF4 import Dataset
from os.path import join

def get_model_characteristics(model):
    """ Returns dictionary with general model information.
    """
    if model == 'ICON':
        model_info = {
            'grid': 'Icosahedral',
            'cumulus_parameterization': 'None',
            'boundary_layer_parameterization': 'TKE-like (with additional progn. equ.)', 
            'fractional_cloudiness_parameterization': True,
        }
    elif model == 'NICAM':
        model_info = {
            'grid': 'Icosahedral',
            'cumulus_parameterization': 'None',
            'boundary_layer_parameterization': 'Diagnostic eddy diffusivity', 
            'fractional_cloudiness_parameterization': False,
        }
    elif model == 'GEOS':
        model_info = {
            'grid': 'Cube',
            'cumulus_parameterization': 'Full',
            'boundary_layer_parameterization': 'Diagnostic eddy diffusivity', 
            'fractional_cloudiness_parameterization': True,
        }        
    elif model == 'SAM':
        model_info = {
            'grid': 'Lat-Lon',
            'cumulus_parameterization': 'None',
            'boundary_layer_parameterization': 'Smagorinsky-like', 
            'fractional_cloudiness_parameterization': False,
        } 
    elif model == 'UM':
        model_info = {
            'grid': 'Lat-Lon',
            'cumulus_parameterization': 'Shallow',
            'boundary_layer_parameterization': 'Diagnostic eddy diffusivity', 
            'fractional_cloudiness_parameterization': True,
        } 
    elif model == 'IFS':
        model_info = {
            'grid': 'Octosahedral',
            'cumulus_parameterization': 'Shallow',
            'boundary_layer_parameterization': 'Diagnostic eddy diffusivity', 
            'fractional_cloudiness_parameterization': True,
        }
    return model_info

        
def read_var(infile, model, varname, specific_names=False):
    """ Reads in one variable from a given netCDF file.
    
    Parameters:
        infile (str): Path to input file
        model (str): Name of model that the data belongs to
        varname (str): Name of netCDF variable to read in
        specific_names (bool): True if modelspecific variable names should be used
        (e.g. if model data is already vertically interpolated)
    
    Returns:
        float or ndarray: Data array with variable
    """
    with Dataset(infile) as ds:
        if not specific_names:
            var = ds.variables[varname][:].filled(np.nan)
        else:
            if model == 'SAM' and varname == 'PRES':
                pres_mean = ds.variables['p'][:].filled(np.nan)
                pres_pert = ds.variables['PP'][:].filled(np.nan)
                var = np.zeros((pres_pert.shape[1], pres_pert.shape[2], pres_pert.shape[3]))
                for i in range(pres_pert.shape[2]):
                    for j in range(pres_pert.shape[3]):
                        var[:, i, j] = pres_mean * 1e2 + pres_pert[0, :, i, j]
            else:
                model_varname = processing_tools.get_modelspecific_varnames(model)[varname]
                var = ds.variables[model_varname][:].filled(np.nan)
        if model == 'MPAS' and len(var.shape) == 4 and varname != 'RH': 
            var = var.transpose((0, 3, 1, 2))
            
    return var

def read_var_timestep(infile, model, varname, timestep, specific_names=False):
    """ Reads in one variable from a given netCDF file.
    
    Parameters:
        infile (str): Path to input file
        model (str): Name of model that the data belongs to
        varname (str): Name of netCDF variable to read in
        timestep (int): Timestep of field to read in
        specific_names (bool): True if modelspecific variable names should be used
        (e.g. if model data is already vertically interpolated)
        
    Returns:
        float or ndarray: Data array with variable
    """
    with Dataset(infile) as ds:
        if not specific_names:
            var = ds.variables[varname][timestep].filled(np.nan)
        else:
            if model == 'SAM' and varname == 'PRES':
                pres_mean = ds.variables['p'][timestep].filled(np.nan)
                pres_pert = ds.variables['PP'][timestep].filled(np.nan)
                var = np.zeros((pres_pert.shape[1], pres_pert.shape[2], pres_pert.shape[3]))
                for i in range(pres_pert.shape[2]):
                    for j in range(pres_pert.shape[3]):
                        var[:, i, j] = pres_mean * 1e2 + pres_pert[0, :, i, j]
            else:
                model_varname = processing_tools.get_modelspecific_varnames(model)[varname]
                var = ds.variables[model_varname][timestep].filled(np.nan)
        if model == 'MPAS' and len(var.shape) == 4 and varname != 'RH': 
            var = var.transpose((0, 3, 1, 2))
            
    return var

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

def calc_IWP(ice_mass_mixing_ratio, temperature, pressure, specific_humidity, height):
    """ Calculate Ice Water Path (IWP) by integrating the product of ice mass mixing ratio and
    the density of moist air over height.
    
    Parameters:
        ice_mass_mixing_ratio (float or ndarray): Mass mixing ratio of ice [kg/kg]
        temperature (float or ndarray): Temperature [K]
        pressure (float or ndarray): Pressure [Pa]
        specific_humidity (float or ndarray): Specific humidity [kg/kg]
        height (1darray): Height vector [m]
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

def calc_IWV(specific_humidity, temperature, pressure, height):
    """ Calculate integrated water vapor (IWV).
    
    Parameters:
        specific_humidity (float or ndarray): Specific humidity [kg/kg]
        temperature (float or ndarray): Temperature [K]
        pressure (float or ndarray): Pressure [Pa]
        height (1darray): Height vector [m]
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

def get_height_array(height, shape):
    """ Create an ndarray of heights from a 1darray.
    
    Parameters:
        height (1darray): height vector 
        shape (tupel): Shape of output ndarray
    """
    if len(shape) > 1:
        shape_hor = shape[1:]
        h = np.repeat(height, np.prod(shape_hor))
        h = np.reshape(h, shape)
    elif len(shape) == 1:
        h = height
    
    return h

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

def get_cold_point_height(height, temperature):
    """ Return the height of the cold point (tropopause) of a given vertical temperature profile.
    
    Parameters:
        height (1D array): height in m
        temperature (1D or 3D array): temperature profile or x-y field of temperature profiles 
            (first dimension corresponds to height)
    """
    tropos_ind = np.where(height < 20000)
    height = height[tropos_ind]
    temperature = temperature[tropos_ind]
    
    cold_point_height = height[np.nanargmin(temperature, axis=0)]
    
    return cold_point_height
    