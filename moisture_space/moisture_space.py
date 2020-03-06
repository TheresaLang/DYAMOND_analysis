import utils
import numpy as np
from scipy.interpolate import interp1d

class MoistureSpace:
    """ Class storing atmospheric profiles in moisture space. """
    def __init__(self, profile_stats=None, bins=None, levels=None):
        """ Create instance of class MoistureSpace.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)   
        """
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
        
    def value_at_level(self, level):
        """ Returns values of the variable (mean) at a given level and all bins.
        
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
        """
        return None
        
class PercMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally sized bins, i.e. bin statistics have
    been calculated from an equal number of profiles in each bin (percentiles). """
    def __init__(self, profile_stats=None, levels=None, bins=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)   
        """
        super().__init__(profile_stats, levels, bins)
        
    def mean_profile(self, bin_start=None, bin_end=None):
        """
        """
        return None
        
class BinMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally spaced bins, i.e. bin statistics have
    been calculated from different numbers of profiles in each bin."""
    def __init__(self, profile_stats=None, levels=None, bins=None, profile_pdf=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)
            profile_pdf (array): Number of profiles in each bin (same length as ``bins``)
        """
        super().__init__(profile_stats, levels, bins)
        self.profile_pdf = profile_pdf
        
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
    
    def transpose(self):
        """ Transpose all attributes that are ndarrays.
        """
        for a in vars(self):
            s = getattr(self, a)
            if (s is not None) and (not isinstance(s, str)):
                print(s.shape)
                setattr(self, a, s.T)
        
