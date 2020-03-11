import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from importlib import reload

class MoistureSpace:
    """ Class storing atmospheric profiles in moisture space. """
    def __init__(self, name=None, profile_stats=None, bins=None, levels=None):
        """ Create instance of class MoistureSpace.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)   
        """
        self.name = name
        self.profile_stats = profile_stats  # (bins, levels)
        self.variable = profile_stats.variable
        self.levels = levels
        self.bins = bins
        if levels is None:
            self.num_levels = 1
        else:
            self.num_levels = len(self.levels)
        self.num_profiles = len(self.bins)
        
        if self.profile_stats.mean.shape[0] != self.num_profiles:
            self.profile_stats.transpose()
            
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return MoistureSpace(self.name, self.profile_stats, self.bins, self.levels)
    
    @property
    def mean(self):
        """ Returns the mean profiles, i.e. the attribute profile_stats.mean.
        """
        return self.profile_stats.mean
        
    def value_at_level(self, level):
        """ Returns values of the variable (mean) at any given level and all bins.
        Linear interpolation between descrete levels is performed to determine
        the value at the given level.
        
        Parameters:
            level (float): Level
            
        Returns:
            array: Values of variable at ``level``
        """
        values_at_level = np.ones(self.num_profiles) * np.nan
        for p in range(self.num_profiles):
            values_at_level[p] = interp1d(self.levels, self.profile_stats.mean[p])(level)

        return values_at_level
    
    def vertical_mean(self, lower_bnd=None, upper_bnd=None):
        """ Returns mean of variable (mean) over a vertical layer for each bin (weighted with layer width).
        
        Parameters:
            lower_bnd (float): Lower bound of vertical layer (default: None; lowest level is used)
            upper_bnd (float): Upper bound of vertical layer (defualt: None; highest level is used)
            
        Returns:
            array: Mean values of variable in layer for each bin.
        """
        if lower_bnd is None:
            lower_bnd = np.min(self.levels)
        if upper_bnd is None:
            upper_bnd = np.max(self.levels)
        
        layer_ind = np.logical_and(self.levels >= lower_bnd, self.levels <= upper_bnd)
        delta_lev = np.gradient(self.levels)
        
        layer_mean = np.average(self.profile_stats.mean[:, layer_ind], axis=1, weights=delta_lev[layer_ind])
        
        return layer_mean
    
    def interpolate(self, levels):
        """ Interpolate profile statistics vertically to given levels.
        
        Parameters:
            levels (array): Levels to interpolate on
            
        Returns: 
            MoistureSpace: New instance containing interpolated fields.
        """
        ret = self.copy()
        #interpolate
        for s in vars(self.profile_stats):
            field = getattr(self.profile_stats, s)
            if (field is not None) and (not isinstance(field, str)):
                field_interp = np.ones((self.num_profiles, len(levels))) * np.nan
                for p in range(self.num_profiles):
                    field_interp[p] = interp1d(self.levels, field[p], fill_value='extrapolate',\
                                               bounds_error=False)(levels)
                ret.profile_stats.set_attr(s, field_interp)  
        ret.levels = levels
        ret.num_levels = len(levels)
        return ret
        
class PercMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally sized bins, i.e. bin statistics have
    been calculated from an equal number of profiles in each bin (percentiles). """
    def __init__(self, name=None, profile_stats=None, levels=None, bins=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)   
        """
        super().__init__(name, profile_stats, levels, bins)
        
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return PercMoistureSpace(self.name, self.profile_stats, self.bins, self.levels)
        
    def mean_profile(self, bin_start=None, bin_end=None):
        """ Returns the mean over several bins.
        
        Parameters:
            bin_start (float): First bin to use (default: None; first bin is used)
            bin_end (float): Last bin to use (defualt: None; last bin is used)
            
        Returns:
            array: Mean values of variable over all bins between bin_start and bin_end for each level.            
        """
        if bin_start is None:
            bin_start = np.min(self.bins)
        if bin_end is None:
            bin_end = np.max(self.bins)
        
        bin_ind = np.logical_and(self.bins >= bin_start, self.bins <= bin_end)
        bin_mean = np.mean(self.profile_stats.mean[bin_ind], axis=0)
        
        return bin_mean
    
    def area_mean(self, bin_start=None, bin_end=None, lower_bnd=None, upper_bnd=None):
        """ Calculate the average over a certain area in moisture space.
        
        Parameters:
            bin_start (float): First bin to use (default: None; first bin is used)
            bin_end (float): Last bin to use (defualt: None; last bin is used)
        """
        if bin_start is None:
            bin_start = np.min(self.bins)
        if bin_end is None:
            bin_end = np.max(self.bins)            
        if lower_bnd is None:
            lower_bnd = np.min(self.levels)
        if upper_bnd is None:
            upper_bnd = np.max(self.levels)
        
        layer_ind = np.logical_and(self.levels >= lower_bnd, self.levels <= upper_bnd)
        bin_ind = np.logical_and(self.bins >= bin_start, self.bins <= bin_end)
        delta_lev = np.gradient(self.levels)
        layer_mean = np.average(self.profile_stats.mean[:, layer_ind], axis=1, weights=delta_lev[layer_ind])       
        area_mean = np.mean(layer_mean[bin_ind], axis=0)
        
        return area_mean
        
class BinMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally spaced bins, i.e. bin statistics have
    been calculated from different numbers of profiles in each bin."""
    def __init__(self, name=None, profile_stats=None, levels=None, bins=None, profile_pdf=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)
            profile_pdf (array): Number of profiles in each bin (same length as ``bins``)
        """
        super().__init__(name, profile_stats, levels, bins)
        self.profile_pdf = profile_pdf
        
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return BinMoistureSpace(self.name, self.profile_stats, self.bins, self.levels, self.profile_pdf)
        
    def remove_empty_bins(self, number_threshold):
        """ Remove all bins with less profiles than ``number_threshold``.
        
        Parameters:
            number_threshold (int): Minimum number of profiles in a bin
        """
        pass
        empty_bins = np.where(self.profile_pdf <= number_threshold)[0]
        self.profile_stats.remove_bins(empty_bins)
                
class ProfileStats:
    """ Class containing statistics of binned profiles. """
    def __init__(self, variable=None, mean=None, median=None, std=None, minimum=None, maximum=None):
        """ Create instance of class ProfileStats.
        
        Parameters:
            variable (str): Variable name
            mean (ndarray): Mean of variable at each bin and level
            median (ndarray): Median of variable at each bin and level
            std (ndarray): Standard deviation of variable at each bin and level
            minimum (ndarray): Minimum of variable at each bin and level
            maximum (ndarray): Maximum of variable at each bin and level
        """
        self.variable = variable
        self.mean = mean
        self.median = median
        self.std = std
        self.min = minimum
        self.max = maximum
        
    @classmethod
    def from_dict(cls, dictionary, variable):
        """ Create instance of class ProfileStats from a dictionary.
        
        Parameters:
            dictionary (dict): Nested dictionary with keys corresponding to statistics and variables
            variable (str): Variable name
        """
        ret = cls()
        ret.variable = variable
        ret.mean = dictionary['mean'][variable]
        ret.median = dictionary['median'][variable]
        ret.std = dictionary['std'][variable]
        ret.min = dictionary['min'][variable]
        ret.max = dictionary['max'][variable]
        
        return ret
    
    def copy(self):
        """ Create a new instance of ProfileStats, with the same data as this instance.
        """
        return ProfileStats(self.variable, self.mean, self.median, self.std, self.min, self.max)
    
    def set_attr(self, attribute, value):
        """ Set an attribute. """
        setattr(self, attribute, value)
                                    
    def transpose(self):
        """ Transpose all attributes that are ndarrays.
        """
        for a in vars(self):
            s = getattr(self, a)
            if (s is not None) and (not isinstance(s, str)):
                setattr(self, a, s.T)
                
    def remove_bins(self, bin_ind):
        """ Set statistics of specified bins to NaN.
        
        Parameters:
            bin_ind (array or list): Indices of bins to remove
        """
        for a in vars(self):
            s = getattr(self, a)
            if (s is not None) and (not isinstance(s, str)):
                s_new = s.copy()
                s_new[bin_ind] = np.nan
                setattr(self, a, s_new)
                
class MoistureSpaceCollection:
    """ Class to calculate statistics over several moisture spaces, e.g. from several models.
    """
    def __init__(self, moisture_spaces):
        """ Create instance of class MoistureSpaceCollection.
        
        Parameters:
            moisture_spaces (list of MoistureSpace): List of MoistureSpace instances with 
                identical bins and levels. 
        """
        self.moisture_spaces = moisture_spaces
        self.num_spaces = len(moisture_spaces)
        self.bins = moisture_spaces[0].bins
        self.levels = moisture_spaces[0].levels
        self.num_bins = len(self.bins)
        self.num_levels = len(self.levels)
        
    @property
    def space_array(self):
        """ Concatenate all moisture spaces (means) of this collection to one array.
        
        Returns:
            3darray: Array containing all moisture spaces of this collection
        """
        space_array = np.ones((self.num_spaces, self.num_bins, self.num_levels)) * np.nan
        for n in range(self.num_spaces):
            space_array[n] = self.moisture_spaces[n].mean
        
        return space_array
    
    @property
    def variability(self):
        """ Calculate variability among different moisture spaces.
        """ 
        return np.std(self.space_array, axis=0)
    
    def calc_EOFs(self):
        """ Calculate Empirical Orthogonal Functions (EOFs) for the collection of different 
        moisture spaces.
        """
        reload(utils)
        reshaped_space_array = self.space_array.reshape((self.num_spaces, self.num_bins * self.num_levels))
        eofs, variability_frac, expansion_coeffs = utils.eofs(reshaped_space_array)
        eofs = eofs.reshape(self.num_bins, self.num_levels, self.num_bins * self.num_levels).transpose(2, 0, 1)
        
        self.eofs = eofs
        self.variability_frac = variability_frac
        self.expansion_coeffs = expansion_coeffs
        
        return eofs, variability_frac, expansion_coeffs
    
    def recunstruct_from_EOFs(self, num_EOFs):
        """
        """
        eofs = self.eofs.transpose(1, 2, 0).reshape(self.num_bins * self.num_levels, self.num_bins * self.num_levels)
        
        #0: model 0
        #rep = np.matmul(np.expand_dims(expansion_coeffs['RH'][0], 1), np.expand_dims(eofs['RH'][0].ravel(), 0))
        #plt.imshow(np.real(rep[5].reshape(100, 50)))
        
        
        
       
        
