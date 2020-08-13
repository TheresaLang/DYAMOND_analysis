import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from importlib import reload
from scipy.stats import pearsonr, spearmanr

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
        self.num_profiles = len(self.bins)
        
        
        if self.profile_stats.mean.shape[0] != self.num_profiles:
            self.profile_stats.transpose()
            
        if levels is None:
            self.num_levels = 1
        else:
            self.num_levels = len(self.levels)
        
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return MoistureSpace(self.name, self.profile_stats.copy(), self.bins, self.levels)
    
    @classmethod
    def from_mean(cls, mean, variable, name=None, bins=None, levels=None):
        """ Create MoistureSpace from only a mean field.
        """
        profile_stats = ProfileStats(variable=variable, mean=mean.T)
        ret = cls(name, profile_stats, bins, levels)
        
        return ret
    
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
    
    def interpolate(self, levels, kind='linear'):
        """ Interpolate profile statistics vertically to given levels.
        
        Parameters:
            levels (1darray): Levels to interpolate on
            kind (str): Type of interpolation (e.g. 'linear' or'cubic')
            
        Returns: 
            MoistureSpace: New instance containing interpolated fields.
        """
        ret = self.copy()
        #interpolate
        for s in vars(self.profile_stats):
            field = getattr(self.profile_stats, s)
            if (field is not None) and (not isinstance(field, str)):
                field_interp = np.empty((self.num_profiles, len(levels)))
                for p in range(self.num_profiles):
                    field_interp[p] = interp1d(self.levels, field[p], fill_value='extrapolate',\
                                               bounds_error=False, kind=kind)(levels)
                ret.profile_stats.set_attr(s, field_interp)  
        ret.levels = levels
        ret.num_levels = len(levels)
        return ret
    
    def change_levels(self, new_levels, new_levels_interp, kind='linear'):
        """ Interpolate profile statistics to new levels of a different variable (e.g. pressure, temperature)
        
        Parameters:
            new_levels (2darray): Variable for new levels at old levels
            new_levels_interp (1darray): New levels to interpolate on
            kind (str): Type of interpolation (e.g. 'linear' or'cubic')
        """
        ret = self.copy()
        
        for s in vars(self.profile_stats):
            field = getattr(self.profile_stats, s)
            if (field is not None) and (not isinstance(field, str)):
                field_interp = np.empty((self.num_profiles, len(new_levels_interp)))
                for p in range(self.num_profiles):
                    field_interp[p] = interp1d(np.log(new_levels[p]), field[p], fill_value='extrapolate',\
                                               bounds_error=False, kind=kind)(np.log(new_levels_interp))
                ret.profile_stats.set_attr(s, field_interp)
        
        ret.levels = new_levels_interp
        ret.num_levels = len(ret.levels)
        
        return ret
    
    def select_region(self, bin_start=None, bin_end=None, lower_bnd=None, upper_bnd=None):
        """ Select a specified region from the moisture space.
        """ 
        if bin_start is None:
            bin_start = np.min(self.bins)
        if bin_end is None:
            bin_end = np.max(self.bins)            
        if lower_bnd is None:
            lower_bnd = np.min(self.levels)
        if upper_bnd is None:
            upper_bnd = np.max(self.levels)
        
        bin_ind = np.logical_and(self.bins >= bin_start, self.bins <= bin_end)
        layer_ind = np.logical_and(self.levels >= lower_bnd, self.levels <= upper_bnd)
        
        for s in vars(self.profile_stats):
            field = getattr(self.profile_stats, s)
            if (field is not None) and (not isinstance(field, str)):
                field_region = field[bin_ind][:, layer_ind]
                setattr(self.profile_stats, s, field_region)
       
        self.bins = self.bins[bin_ind]
        self.num_profiles = len(self.bins)
        self.levels = self.levels[layer_ind]
        self.num_levels = len(self.levels)
        
    def remove_levels(self, level_ind):
        """ Remove levels with indices specified in level_ind.
        
        Parameters:
            level_ind (array): indices of levels to remove
        """
        remaining_levels = [i for i in range(self.num_levels) if i not in level_ind]
        self.levels = self.levels[remaining_levels]
        self.num_levels = len(self.levels)
        self.profile_stats.remove_levels(level_ind)
        
class PercMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally sized bins, i.e. bin statistics have
    been calculated from an equal number of profiles in each bin (percentiles). """
    def __init__(self, name=None, profile_stats=None, bins=None, levels=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)   
        """
        super().__init__(name, profile_stats, bins, levels)
        
        # remove levels with nans
        if self.levels is not None:
            nan_levels = np.where(np.all(np.isnan(self.profile_stats.mean), axis=0))[0]
            if len(nan_levels) > 0:
                remaining_levels = [i for i in range(len(self.levels)) if i not in nan_levels]
                self.levels = self.levels[remaining_levels]
                self.profile_stats.remove_levels(nan_levels)
        
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return PercMoistureSpace(self.name, self.profile_stats.copy(), self.bins, self.levels)
        
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
    
    def reduced_bins(self, num_bins):
        """ Returns an array with a reduced number of bins. The number of bins
        is reduced by combining/averaging several bins into one. 
        
        Parameters:
            num_bins (int): Number of bins 
        """
        
        splitted_array = np.array_split(self.profile_stats.mean, num_bins)
        averaged_array = np.array([np.mean(a, axis=0) for a in splitted_array])
        
        return averaged_array
    
    def complete_to_TOA(self, aux_levels, aux_profile):
        """ Complete profiles to the top of the atmosphere using a given atmospheric profile (e.g. standard profile). A cubic interpolation is used for the transition.
        
        Parameters: 
            aux_levels:
            aux_profiles:
        """
        
        levels_new = np.hstack((self.levels, aux_levels))
        aux_profiles = np.repeat(np.expand_dims(aux_profile, 0), self.num_profiles, axis=0)
        mean_new = np.hstack((self.profile_stats.mean, aux_profiles))
        
        ret = PercMoistureSpace.from_mean(mean_new, self.variable, self.name, self.bins, levels_new)
        return ret
        
class BinMoistureSpace(MoistureSpace):
    """ Class storing mean atmospheric profiles in equally spaced bins, i.e. bin statistics have
    been calculated from different numbers of profiles in each bin."""
    def __init__(self, name=None, profile_stats=None, bins=None, levels=None, profile_pdf=None):
        """ Create instance of class PercMoistureSpace,.
        
        Parameters:
            profile_stats (ProfileStats): Statistics of profiles in bins (mean, standard deviation etc.)
            bins (array): Bins (x-coordinate of moisture space, e.g. integrated water vapour)
            levels (array): Vertical levels (y-coordinate of moisture space, e.g. height)
            profile_pdf (array): Number of profiles in each bin (same length as ``bins``)
        """
        super().__init__(name, profile_stats, bins, levels)
        self.profile_pdf = profile_pdf
        
    def copy(self):
        """ Create a new instance of MoistureSpace, with the same data as this instance.
        """
        return BinMoistureSpace(self.name, self.profile_stats.copy(), self.bins, self.levels, self.profile_pdf)
        
    def remove_empty_bins(self, number_threshold):
        """ Remove all bins with less profiles than ``number_threshold``.
        
        Parameters:
            number_threshold (int): Minimum number of profiles in a bin
        """
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
        ret.mean = dictionary['mean'][variable].T
        
        try:
            ret.median = dictionary['median'][variable].T
            ret.std = dictionary['std'][variable].T
            ret.min = dictionary['min'][variable].T
            ret.max = dictionary['max'][variable].T
        except:
            pass
        
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
                
    def flip(self):
        """ Flip all attributes that are ndarrays
        """
        
        for a in vars(self):
            s = getattr(self, a)
            if (s is not None) and (not isinstance(s, str)):
                setattr(self, a, np.flipud(s))
                
    def remove_levels(self, level_ind):
        """ Remove statistics of specified levels.
        
        Parameters:
            level_ind (array or list): Indices of levels to remove
        """
        for a in vars(self):
            s = getattr(self, a)
            if (s is not None) and (not isinstance(s, str)):
                s_new = s.copy()
                s_new = np.delete(s_new, level_ind, axis=1)
                setattr(self, a, s_new)
                
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
                
class MoistureSpaceSeries:
    """ Class to calculate statistics over several moisture spaces of the same variable.
    """
    def __init__(self, moisture_spaces, remove_nans=False):
        """ Create instance of class MoistureSpaceSeries.
        
        Parameters:
            moisture_spaces (list of MoistureSpace): List of MoistureSpace instances with 
                identical bins and levels. 
        """
        self.moisture_spaces = moisture_spaces
        self.num_spaces = len(moisture_spaces)
        self.variable = moisture_spaces[0].variable
        self.bins = moisture_spaces[0].bins
        self.levels = moisture_spaces[0].levels
        self.num_bins = len(self.bins)
        self.num_levels = len(self.levels)
        
        # combine all moisture spaces of this series in one array:
        space_array = np.ones((self.num_spaces, self.num_bins, self.num_levels)) * np.nan
        for n in range(self.num_spaces):
            space_array[n] = self.moisture_spaces[n].mean
        
        if remove_nans:
            # cut away NaNs
            nan_ind = np.any(np.any(np.isnan(space_array), axis=0), axis=1)
            space_array = space_array[:, ~nan_ind]
            self.bins = self.bins[~nan_ind]
            self.num_bins = len(self.bins)
        self.space_array = space_array
        
#    @property
#    def space_array(self):
#        """ Concatenate all moisture spaces (means) of this collection to one array.
#        
#        Returns:
#            3darray: Array containing all moisture spaces of this collection
#        """
#        space_array = np.ones((self.num_spaces, self.num_bins, self.num_levels)) * np.nan

#        for n in range(self.num_spaces):
#            space_array[n] = self.moisture_spaces[n].mean
        
#        # cut away NaNs
#        nan_ind = np.any(np.isnan(space_array), axis=0)
#        space_array = space_array[:, ~nan_ind]
#        
#        self.bins = self.bins[~nan_ind[:, 0]]
#        self.num_bins = len(self.bins)
#        
#        return space_array
    
    @property
    def variability(self):
        """ Calculate variability among different moisture spaces.
        """ 
        return np.std(self.space_array, axis=0)
    
    @property
    def mean(self):
        """ Calculate mean of different moisture spaces.
        """
        return np.mean(self.space_array, axis=0)
    
    def reduced_bins(self, num_bins):
        """ Returns mean of different moisture spaces with reduced number of bins.
        
        Parameters:
            num_bins (int): Number of reduced bins
        """
        mean_array = self.mean
        
        splitted_array = np.array_split(mean_array, num_bins)
        averaged_array = np.array([np.mean(a, axis=0) for a in splitted_array])
        
        return averaged_array
    
    def interpolate(self, levels):
        """
        """
        moisture_spaces = []
        for space in self.moisture_spaces:
            space_aux = space.copy()
            moisture_spaces.append(space_aux.interpolate(levels))
        
        return MoistureSpaceSeries(moisture_spaces)
        
    def calc_EOFs(self):
        """ Calculate Empirical Orthogonal Functions (EOFs) for the collection of different 
        moisture spaces.
        """
        reshaped_space_array = self.space_array.reshape((self.num_spaces, self.num_bins * self.num_levels))
        eofs, variability_frac, expansion_coeffs = utils.eofs(reshaped_space_array)
        eofs = eofs.reshape(self.num_bins, self.num_levels, self.num_bins * self.num_levels).transpose(2, 0, 1)
        
        self.eofs = eofs
        self.variability_frac = variability_frac
        self.expansion_coeffs = expansion_coeffs
        
        return eofs, variability_frac, expansion_coeffs
    
    def recunstruct_from_EOFs(self, num_EOFs):
        """ Reconstruct fields from a certain number of EOFs.
        
        Parameters:
            num_EOFs: Number of EOFs to use to reconstruct fields.
        """
        rep = np.zeros((self.num_spaces, self.num_bins, self.num_levels))
        r = np.zeros((self.num_spaces, self.num_bins * self.num_levels))
        for n in range(num_EOFs):
            r += np.real(np.matmul(
                np.expand_dims(self.expansion_coeffs[n], 1),
                np.expand_dims(self.eofs[n].ravel(), 0)
            ))
        rep = r.reshape((self.num_spaces, self.num_bins, self.num_levels))
        
        profile_stats = [ProfileStats(variable=self.moisture_spaces[i].variable, mean=rep[i])\
                         for i in range(self.num_spaces)]
        ret = [PercMoistureSpace(name=i, profile_stats=profile_stats[i], bins=self.bins, levels=self.levels)\
              for i in range(self.num_spaces)]
            
        return ret  

class MoistureSpaceSeriesPair:
    """ Class to perfomr statistics over two MoistureSpaceSeries.
    """
    def __init__(self, moisture_space_series_1, moisture_space_series_2):
        """ Create instance of class MoistureSpaceSeriesPair.
        
        Parameters:
            moisture_space_series_1: First MoistureSpaceSeries
            moisture_space_series_2: Second MoistureSpaceSeries
        """
        self.moisture_space_series = [moisture_space_series_1, moisture_space_series_2]
        self.variables = [moisture_space_series_1.variable, moisture_space_series_2.variable]
        self.bins = moisture_space_series_1.bins
        self.num_bins = moisture_space_series_1.num_bins
        self.levels = moisture_space_series_1.levels
        self.num_levels = moisture_space_series_1.num_levels
        
    def correlation(self, corrtype='pearson'):
        """ Calculate correlation coefficient for each point in moisture space.
        """
        corr_coeff = np.zeros((self.num_bins, self.num_levels))
        space_array_0 = self.moisture_space_series[0].space_array
        space_array_1 = self.moisture_space_series[1].space_array
        
        for b in range(self.num_bins):
            for l in range(self.num_levels):
                if corrtype == 'pearson':
                    corr_coeff[b, l], p = pearsonr(space_array_0[:, b, l], space_array_1[:, b, l])
                elif corrtype == 'spearman':
                    corr_coeff[b, l], p = spearmanr(space_array_0[:, b, l], space_array_1[:, b, l])
        
        return corr_coeff        
        
    def perform_SVD(self):
        """ Perform a Singular Value Decomposition (SVD).
        """
        reload(utils)
        space_arrays = [self.moisture_space_series[i].space_array for i in range(2)]
        reshaped_space_arrays = [
            space_arrays[i].reshape(
                (self.moisture_space_series[i].num_spaces, self.num_bins * self.num_levels)
            ) 
            for i in range(2)
        ]
        
        u, s, v, expansion_coeffs_1, expansion_coeffs_2 = utils.svd(reshaped_space_arrays)
        singular_vec1 = u.reshape((self.num_bins, self.num_levels, self.num_bins * self.num_levels)).transpose(2, 0, 1)
        singular_vec2 = v.reshape((self.num_bins * self.num_levels, self.num_bins, self.num_levels))
        frac_variability = np.array([i / np.sum(s) for i in s])
        
        return singular_vec1, singular_vec2, frac_variability, expansion_coeffs_1, expansion_coeffs_2
        
class MoistureSpaceSeriesCollection():
    """ Class to perfomr statistics over several (two or more) MoistureSpaceSeries.
    """
    def __init__(self, moisture_space_series_list):
        """ Create instance of class MoistureSpaceSeriesCollection.
        
        Parameters:
            moisture_space_series_list: List containing MoistureSpaceSeries objects
        """
        self.moisture_space_series = [series for series in moisture_space_series_list]
        self.num_series = len(self.moisture_space_series)
        self.variables = [series.variable for series in moisture_space_series_list]
        self.bins = self.moisture_space_series[0].bins
        self.num_bins = self.moisture_space_series[0].num_bins
        self.levels = self.moisture_space_series[0].levels
        self.num_levels = self.moisture_space_series[0].num_levels
        self.num_spaces = self.moisture_space_series[0].num_spaces
        
    def calc_EOFs(self):
        """ Calculate EOFs explaining co-variability of variables in different MoistureSpaceSeries.
        """
        reshaped_space_array = np.concatenate([series.space_array.reshape((series.num_spaces, series.num_bins * series.num_levels)) for series in self.moisture_space_series], axis=1)
            
        eofs_combined, variability_frac, expansion_coeffs_combined = utils.eofs(reshaped_space_array)
        eofs_split = np.split(eofs_combined, self.num_series, axis=0)

        eofs = [np.reshape(eof, (self.num_bins, self.num_levels, self.num_bins * self.num_levels * self.num_series)).transpose(2, 0, 1) for eof in eofs_split]
        
        expansion_coeffs_split = np.split(expansion_coeffs_combined, self.num_series, axis=0)
                
        self.eofs = eofs
        self.variability_frac = variability_frac
        self.expansion_coeffs = expansion_coeffs_combined
        
        return eofs, variability_frac, expansion_coeffs_split
        
        
        
        
        

        
        
        
       
        
