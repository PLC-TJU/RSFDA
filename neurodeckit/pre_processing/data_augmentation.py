""" 
This module provides functions for data expansion.

Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/21
License: BSD 3-Clause License
"""

# 数据增强模块

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, filtfilt

class TimeWindowDataExpansion(BaseEstimator, TransformerMixin):
    """Time window data expansion class for EEG data.

    This class provides functions for time window data expansion on EEG data. 
    It takes an EEG array and splits it into samples with different window sizes and steps.

    Attributes:
        fs (int): The sampling rate of the EEG array in Hz.
        window_width (float): The width of the window in seconds.
        window_step (float): The step of the window in seconds.

    Methods:
        fit(X, y=None): This method is not used.
        transform(X, y=None): This method splits the EEG array into samples with different
        window sizes and steps.

    Example:
        >>> eeg = np.random.randn(90, 64, 1000) # generate a random EEG array
        >>> label = np.random.randint(0, 2, 90) # generate a random label array
        >>> fs = 250 # set the sampling rate to 250 Hz
        >>> window_width = 1.5 # set the window width to 1.5 seconds
        >>> window_step = 0.1 # set the window step to 0.1 seconds
        >>> da = TimeWindowDataExpansion(fs, window_width, window_step) # initialize the TimeWindowDataExpansion class
        >>> samples = da.fit_transform(eeg) # split the EEG array into samples
        >>> print(samples.shape) # print the shape of the samples
        (540, 90, 64, 375)
    """
    
    def __init__(self, fs=250, window_width=2, window_step=0.2):
        self.fs = fs
        self.window_width = window_width
        self.window_step = window_step
    
    def __repr__(self):
        return "TimeWindowDataExpansion(fs={}, window_width={}, window_step={})".format(
            self.fs, self.window_width, self.window_step
        )
        
    def fit(self, X, y=None):
        """ This method is not used. """
        return self
    
    def transform(self, X, y=None):
        """ Split the EEG array into samples with different window sizes and steps.

        This method takes an EEG array and splits it into samples with different window sizes
        and steps. The window size and step are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at testing time. The output for
           .fit_transform() will be different than using .fit() and 
           .transform() separately.
           
        """ 
        return X

    def fit_transform(self, X, y=None):
        """ Split the EEG array into samples with different window sizes and steps.

        This method takes an EEG array and splits it into samples with different window sizes
        and steps. The window size and step are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Args:
            X (numpy.ndarray): The EEG array to be split. shape: (n_samples, ..., n_timepoints)
            first dimension should be the number of samples, 
            last dimension should be the number of timepoints.  
            
        Returns:
            numpy.ndarray: The augmented samples. shape: (n_windows, n_samples, ..., n_timepoints)
            
        """
        # Check if X is a 3D array
        while len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)
        
        # Convert window_width and window_step from seconds to samples
        width = int(self.window_width * self.fs)
        step = int(self.window_step * self.fs)
        
        # Get the shape of X
        n_samples = X.shape[0]
        n_channels = X.shape[-2]
        n_timepoints = X.shape[-1]
        
        # Initialize an empty list to store the augmented samples
        augmented_samples = []
        
        # Loop through each sample
        for i in range(n_samples):
            # Get the current sample and its label
            sample = X[i]
            
            # Initialize the start and end indices of the window
            start = 0
            end = width
            
            # Loop until the end index exceeds the number of timepoints
            while end <= n_timepoints:
                # Get the current window
                window = sample[..., start:end]
                
                # Append the window to the augmented_samples list
                augmented_samples.append(window)
                
                # Update the start and end indices by adding the step size
                start += step
                end += step
        
        # Convert the list to a numpy array
        augmented_samples = np.array(augmented_samples)
        
        # Reshape augmented_samples to the desired shape
        n_windows = augmented_samples.shape[0] // n_samples
        augmented_samples = augmented_samples.reshape((n_windows, n_samples, n_channels, width))
        
        return augmented_samples

class FilterBankDataExpansion(BaseEstimator, TransformerMixin):  
    """Filter bank data expansion class for EEG data.

    This class provides functions for filter bank data expansion on EEG data. 
    It takes an EEG array and splits it into samples with different filter bank bands.

    Attributes:
        fs (int): The sampling rate of the EEG array in Hz.
        bands (list): The list of filter bank bands to use. Each band is a tuple of the form (low_freq, high_freq).

    Methods:
        fit(X, y=None): This method is not used.
        transform(X, y=None): This method splits the EEG array into samples with different
        filter bank bands.
        get_filter_coeff(): This method returns the filter coefficients for the filter bank.
        filter_data(X): This method applies the filter bank to the input data.

    Example:
        >>> eeg = np.random.randn(90, 64, 1000) # generate a random EEG array
        >>> label = np.random.randint(0, 2, 90) # generate a random label array
        >>> fs = 250 # set the sampling rate to 250 Hz
        >>> bands = [(4, 8), (8, 12), (12, 30)] # set the filter bank bands
        >>> da = FilterBankDataExpansion(fs, bands) # initialize the FilterBankDataExpansion class
        >>> samples = da.fit_transform(eeg) # split the EEG array into samples
        >>> print(samples.shape) # print the shape of the samples
        (3, 90, 64, 1000) # 5 is the number of filter bank bands
    """
    
    def __init__(self, fs=250, bands=[(4, 8), (8, 12), (12, 30)], order=5):
        self.fs = fs
        self.bands = bands
        self.order = order
    
    def __repr__(self):
        return "FilterBankDataExpansion(fs={}, bands={}, order={})".format(
            self.fs, self.bands, self.order
        )
    
    def get_filter_coeff(self, low_freq, high_freq, fs):
        """ Get the filter coefficients for a given frequency range.

        This method takes a low and high frequency and the sampling rate and returns the filter
        coefficients for a Butterworth filter with a passband between the low and high frequencies.

        Args:
            low_freq (float): The low frequency of the passband in Hz.
            high_freq (float): The high frequency of the passband in Hz.
            fs (int): The sampling rate in Hz.

        Returns:
            tuple: The filter coefficients (b, a) for the Butterworth filter.

        """
        nyq = 0.5 * fs
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a
    
    def filter_data(self, data, b, a):
        """ Apply a Butterworth filter to the input data.

        This method takes input data and filter coefficients and applies the Butterworth filter
        to the data.

        Args:
            data (numpy.ndarray): The input data to be filtered.
            b (numpy.ndarray): The numerator coefficients of the Butterworth filter.
            a (numpy.ndarray): The denominator coefficients of the Butterworth filter.

        Returns:
            numpy.ndarray: The filtered data.

        """
        filtered_data = filtfilt(b, a, data, axis=-1)
        return filtered_data

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """ Split the EEG array into samples with different filter bank bands.

        This method takes an EEG array and splits it into samples with different filter bank bands.
        The filter bank bands are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at testing time. The output for
           .fit_transform() will be different than using .fit() and 
           .transform() separately.
           
        """ 
        return X

    def fit_transform(self, X, y=None):
        """ Split the EEG array into samples with different filter bank bands.

        This method takes an EEG array and splits it into samples with different filter bank bands.
        The filter bank bands are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Args:
            X (numpy.ndarray): The EEG array to be split. shape: (n_samples, ..., n_timepoints)
            first dimension should be the number of samples, 
            last dimension should be the number of timepoints.  
            
        Returns:
            numpy.ndarray: The augmented samples. shape: (n_bands, n_samples, ..., n_timepoints)
            
        """
        # Check if X is a 3D array
        while len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)
        
        # Get the shape of X
        n_samples = X.shape[0]
        n_channels = X.shape[-2]
        n_timepoints = X.shape[-1]
        
        # Initialize an empty list to store the augmented samples
        augmented_samples = []
        
        # Loop through each sample
        for i in range(n_samples):
            # Get the current sample and its label
            sample = X[i]
            
            # Loop through each band
            for band in self.bands:
                # Get the low and high frequencies of the current band
                low_freq, high_freq = band
                
                # Get the filter coefficients for the current band
                b, a = self.get_filter_coeff(low_freq, high_freq, self.fs)
                
                # Filter the current sample using the current filter coefficients
                filtered_sample = self.filter_data(sample, b, a)
                
                # Append the filtered sample to the augmented_samples list
                augmented_samples.append(filtered_sample)
        
        # Convert the list to a numpy array
        augmented_samples = np.array(augmented_samples)     

        # Reshape augmented_samples to the desired shape
        n_bands = len(self.bands)
        augmented_samples = augmented_samples.reshape((n_bands, n_samples, n_channels, n_timepoints))        
        
        return augmented_samples
    
    
# 时间滑动窗口数据扩充类(与新版不同，新版是返回的窗口数*样本数*通道数*窗口长度的4D数组，
# 旧版是返回的[窗口数*样本数]*通道数*窗口长度的3D数组)      
class TimeWindowDataExpansion_old(BaseEstimator, TransformerMixin):
    """Time window data expansion class for EEG data.

    This class provides functions for time window data expansion on EEG data. 
    It takes an EEG array and splits it into samples with different window sizes and steps.

    Attributes:
        fs (int): The sampling rate of the EEG array in Hz.
        window_width (float): The width of the window in seconds.
        window_step (float): The step of the window in seconds.

    Methods:
        fit(X, y=None): This method is not used.
        transform(X, y=None): This method splits the EEG array into samples with different
        window sizes and steps.

    Example:
        >>> eeg = np.random.randn(90, 64, 1000) # generate a random EEG array
        >>> label = np.random.randint(0, 2, 90) # generate a random label array
        >>> fs = 250 # set the sampling rate to 250 Hz
        >>> window_width = 1.5 # set the window width to 1.5 seconds
        >>> window_step = 0.1 # set the window step to 0.1 seconds
        >>> da = DataExpansion(fs, window_width, window_step) # initialize the DataExpansion class
        >>> samples, labels = da.fit_transform(eeg, label) # split the EEG array into samples
        >>> print(samples.shape) # print the shape of the samples
        (4050, 64, 150)
    """
    
    def __init__(self, fs=250, window_width=2, window_step=0.2):
        self.fs = fs
        self.window_width = window_width
        self.window_step = window_step
    
    def __repr__(self):
        return "TimeWindowDataExpansion(fs={}, window_width={}, window_step={})".format(
            self.fs, self.window_width, self.window_step
        )

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """ Split the EEG array into samples with different window sizes and steps.

        This method takes an EEG array and splits it into samples with different window sizes
        and steps. The window size and step are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at testing time. The output for
           .fit_transform() will be different than using .fit() and 
           .transform() separately.
           
        """ 
        return X

    def fit_transform(self, X, y=None):
        """ Split the EEG array into samples with different window sizes and steps.

        This method takes an EEG array and splits it into samples with different window sizes
        and steps. The window size and step are specified in the constructor of the class.
        
        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Args:
            X (numpy.ndarray): The EEG array to be split. shape: (n_samples, ..., n_timepoints)
            first dimension should be the number of samples, 
            last dimension should be the number of timepoints.  
            
        Returns:
            numpy.ndarray: The augmented samples. shape: (n_samples * n_windows, ..., n_timepoints)
            
        """
        # check if X is a 3D array
        while len(X.shape)< 3:
            X = np.expand_dims(X, axis=0)
        # convert window_width and window_step from seconds to samples
        width = int(self.window_width * self.fs)
        step = int(self.window_step * self.fs)
        # get the shape of X
        n_samples = X.shape[0]
        n_timepoints = X.shape[-1]
        # initialize an empty list to store the augmented samples
        augmented_samples = []
        # loop through each sample
        for i in range(n_samples):
            # get the current sample and its label
            sample = X[i]
            # initialize the start and end indices of the window
            start = 0
            end = width
            # loop until the end index exceeds the number of timepoints
            while end <= n_timepoints:
                # get the current window
                window = sample[..., start:end]
                # append the window to the augmented_samples lists
                augmented_samples.append(window)        
                # update the start and end indices by adding the step size
                start += step
                end += step
        # convert the list to a numpy array and return it
        return np.array(augmented_samples)


