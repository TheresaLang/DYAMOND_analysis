import numpy as np
import copy

class Modeldata:
    """ Instances of Modeldata contain data from one model. Data is stored as DataField objects.
    For initialization of a Modeldata object, 3D fields (global fields) of variables have to be passed.
    """
    def __init__(self, model, dimensions, dimensionvars, global_fields):
        """ To initialize an object of the class Modeldata, the following arguments are needed:
        
            model (str): Name of model
            dimensions (list): Names of dimensions of the 3D global fields passed for initialization 
                (e.g. ('height', 'latitude', 'longitude')) 
            dimensionvars (dictionary): Dictionary with keys equal to the entries of dimensions, which contains
                vectors with the corresponding variables.
            global_fields (dictionary of 3D arrays): Dictionary containing 3D global fields of different model variables. 
                Dictionary keys correspond to the names of these variables.
        """
        self.model = model
        self.variables = list(global_fields.keys())
        self.dimensions = dimensions
        self.dimensionvars = dimensionvars
        self.global_fields = dict.fromkeys(self.variables)
        
        for var in self.variables:
            self.global_fields[var] = VariableField(
                self.dimensions,
                self.dimensionvars, 
                global_fields[var]
            )
                
        self.zonal_means = None
        self.tropical_mean_profiles = None
    
    def add_global_field(self, field, variable):
        """ Add a global field of a specific variable to the modeldata object.
        
        Parameters:
            field (3D array): 3D data field (must have the same dimensions as the global fields
                that are already contained in the object.)
            variable (str): Name of variable
        """
        self.global_fields[variable] = VariableField(
            self.dimensions,
            self.dimensionvars,
            field
        )
        self.variables.append(variable)
    
    def get_global_fields(self):
        """ Return global fields of modeldata object (as VariableField objects).
        """
        return self.global_fields
        
    def calc_zonal_means(self):
        """ Calculate zonal means of global fields.
        """

        zonal_means = dict.fromkeys(self.variables)
        
        for var in self.variables:
            zonal_means[var] = self.global_fields[var].calc_average('longitude')
                        
        self.zonal_means = zonal_means
    
    def get_zonal_means(self):
        """ Returns zonal means of global fields (as VariableField objects).
        """
        if self.zonal_means is None:
            self.calc_zonal_means()
        
        return self.zonal_means
    
    def calc_tropical_mean_profiles(self):
        """ Calculate mean tropical profiles of global fields.
        """
        tropical_mean_profiles = dict.fromkeys(self.variables)
        zonal_means = self.get_zonal_means()
        
        for var in self.variables:
            tropical_mean_profiles[var] = zonal_means[var].select_subfield('latitude', [-30, 30]).calc_average('latitude')
            
        self.tropical_mean_profiles = tropical_mean_profiles
    
    def get_tropical_mean_profiles(self):
        """ Return mean tropical profiles of global fields (as VariableFields objects).
        """
        if self.tropical_mean_profiles is None:
            self.calc_tropical_mean_profiles()
        
        return self.tropical_mean_profiles
        
        
class VariableField:
    """ Instances of VariableField contain data fields of one variable and
    corresponding information on the dimensions of the field. 
    """
    def __init__(self, dimensions, dimensionvars, data):
        """ To initialize a Variable Field, the following parameters have to be passed:
        
        dimensions (list of str): Names of dimensions of the field passed for initialization 
             (e.g. ('height', 'latitude', 'longitude')) 
        dimensionvars (dict): Dictionary with keys equal to the entries of the parameter dimensions, 
            which contains vectors with the corresponding variables.
        data (ndarray): Data field (dimensions must match the entries of the parameter dimensions)
        """
        self.dimensions = dimensions
        self.dimensionvars = dimensionvars
        self.data = data
        
    def calc_average(self, dimension):
        """ Average the field along a given dimension.
        
        Parameters:
            dimension (str): dimension to average along
        
        Returns:
            VariableField: averaged field
        """
        average_axis = self.dimensions.index(dimension)
        dimensions_average = self.get_reduced_dimensions([dimension])
        dimensionvars_average = self.get_reduced_dimensionvars([dimension])
        average = VariableField(
            dimensions_average,
            dimensionvars_average,
            np.squeeze(np.nanmean(self.data, axis=average_axis))
        )
        return average
    
    def select_subfield(self, dimension, bounds):
        """ Select a subfield along a given dimension.
        
        Parameters:
            dimension (str): Name of dimension from which to take a subset (e.g. 'latitude')
            bnds (list): Lower and upper boundary for selection (e.g. [-30, 30])
            
        Returns:
            VariableField: Selected subfield
        """
        selection_axis = self.dimensions.index(dimension)
        selection_ind = np.where(
            np.logical_and(
                self.dimensionvars[dimension] >= bounds[0],
                self.dimensionvars[dimension] <= bounds[1]
            )
        )[0]

        subfield = VariableField(
            self.dimensions,
            self.dimensionvars,
            np.take(self.data, selection_ind, selection_axis)
        )
        
        return subfield
    
    def get_reduced_dimensions(self, dimensions_to_remove):
        """ Returns reduced list of dimensions.
        
        Parameters:
            dimensions_to_remove (list of str): List of names of dimensions to remove from the original list.
            
        Returns:
            list of str: Reduced list of dimensions
        """
        reduced_dimensions = self.dimensions.copy()
        for dim in dimensions_to_remove:
            reduced_dimensions.remove(dim)
        
        return reduced_dimensions
    
    def get_reduced_dimensionvars(self, dimensionvars_to_remove):
        """ Returns reduced dictionary of dimension variables.
        
        Parameters:
            dimensionvars_to_remove (list of str): List of names of dimensions to remove from the
                dictionary containing the dimension variables.
                
        Returns:
            dict: reduced dictionary containing dimension variables
        """
        reduced_dimensionvars = dict(self.dimensionvars)
        for dim in dimensionvars_to_remove:
            del reduced_dimensionvars[dim]
            
        return reduced_dimensionvars

