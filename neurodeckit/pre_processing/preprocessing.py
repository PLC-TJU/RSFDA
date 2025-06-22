"""
Pre-processing
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/21
License: All rights reserved
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.channelselection import FlatChannelRemover

from . import Downsample, ChannelSelector, BandpassFilter, TimeWindowSelector, RemoveMean
from . import RiemannChannelSelector, CSPChannelSelector
from . import TimeWindowDataExpansion
from . import RSF

class Pre_Processing(BaseEstimator, TransformerMixin): 
    
    """
    Pre-processing class for data preprocessing.

    Attributes:
        steps (list): a list of tuples containing the name and the estimator object for each step in the pre-processing pipeline.
        process (Pipeline): the pre-processing pipeline.
        
        # downsampling parameters
        fs_new (int or None): the new sampling frequency after downsampling.
        fs_old (int or None): the original sampling frequency before downsampling.
        
        # channel selection parameters
        channels (list or None): the list of channels to select.
        
        # time window selection parameters
        start_time (float or None): the start time of the time window. seconds.
        end_time (float or None): the end time of the time window. seconds.
        
        # bandpass filter parameters
        lowcut (float or None): the lower cutoff frequency of the bandpass filter. 
        highcut (float or None): the upper cutoff frequency of the bandpass filter.
        order (int): the order of the bandpass filter. 
        filter_type (str): the type of the bandpass filter. default is 'butter'. options: 'butter', 'cheby1'
        
        # channel selection plus parameters
        cs_method (str or None): the method of channel selection. default is None. 
                                 options: 'riemann-cs', 'csp-cs', 'default-rsf', 'csp-rsf', 'riemann-csp-rsf', 'cspf'
        nelec (int): the number of electrodes for Riemann channel selection. default is 16.
        
        # data augmentation parameters
        aug_method (str or None): the method of data augmentation. default is None. options: 'time_window'
        window_width (float): the width of the time window for time window data augmentation. seconds.
        window_step (float): the step of the time window for time window data augmentation. seconds.    
        
    Methods:
        fit(X, y=None): fit the pre-processing pipeline.
        transform(X): transform the input data using the pre-processing pipeline.   
    
    Example:
        ```python
        from preprocessing import Pre_Processing
        
        # initialize the pre-processing object
        pre_processor = Pre_Processing(fs_new=200, fs_old=1000, channels=[0, 1, 2], start_time=0.5, end_time=3, lowcut=5, highcut=30)
        
        # fit the pre-processing pipeline
        pre_processor.fit(X, y)
        
        # transform the input data
        X_transformed = pre_processor.transform(X)
        ```
    """
    
    def __init__(self, 
                 fs_new=None, fs_old=None,          # downsampling
                 channels=None,                     # channel selection
                 start_time=None, end_time=None,    # time window selection
                 lowcut=None, highcut=None, order=5, filter_type='butter', # bandpass filter    
                 cs_method=None, nelec=10,          # channel selection plus
                 aug_method=None, window_width=1.5, window_step=0.5,  # data augmentation
                 **kwargs                           # optional parameters
                 ):
        
        self.steps = []
        self.process = None
        self.compat_flag = True  # compatibility, if True, the final pipeline (self.process) will be compatible with other sklearn pipeline
        self.memory = kwargs.get('memory', None)
        self.kwargs = kwargs
        
        # downsampling parameters
        self.fs_new = fs_new
        self.fs_old = fs_old
        
        # channel selection parameters
        self.channels = channels
        
        # bandpass filter parameters
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order  # bandpass filter order
        self.filter_type = filter_type  # bandpass filter type
        
        # channel selection parameters plus
        self.cs_method = cs_method  # channel selection method
        self.nelec = nelec  # number of electrodes for Riemann channel selection
        
        # time window selection parameters
        self.start_time = start_time
        self.end_time = end_time
        
        ## optional parameters
        # data augmentation parameters
        self.aug_method = aug_method  # data augmentation method
        self.window_width = window_width # window width for time window data augmentation
        self.window_step = window_step  # window step for time window data augmentation
        
        ## initialize the pre-processing steps
        
        # downsampling
        if self.fs_new is not None and self.fs_old is not None:
            self.steps.append(('downsample', Downsample(fs_new=self.fs_new, fs_old=self.fs_old)))
        # channel selection
        if self.channels is not None:
            self.steps.append(('channel_selector', ChannelSelector(channels=self.channels)))
        self.steps.append(('flat_channel_remover', FlatChannelRemover()))
        # bandpass filter
        if self.lowcut is not None and self.highcut is not None:
            self.steps.append(('bandpass_filter', BandpassFilter(fs=self.fs_new, 
                                                                 lowcut=self.lowcut, 
                                                                 highcut=self.highcut, 
                                                                 order=self.order, 
                                                                 filter_type=self.filter_type
                                                                 )))
        # channel selection plus
        if self.cs_method is not None:
            if self.cs_method == 'riemann-cs':
                self.steps.append(('channel_selector_plus', RiemannChannelSelector(nelec=self.nelec)))
            elif self.cs_method == 'csp-cs':
                self.steps.append(('channel_selector_plus', CSPChannelSelector(nelec=self.nelec)))  
            elif self.cs_method in ['default-rsf', 'rsf']:
                self.steps.append(('spatial_filter', RSF(dim=self.nelec, method='default')))
            elif self.cs_method == 'csp-rsf':
                self.steps.append(('spatial_filter', RSF(dim=self.nelec, method='csp')))
            elif self.cs_method == 'riemann-csp-rsf':
                self.steps.append(('spatial_filter', RSF(dim=self.nelec, method='riemann-csp')))
            elif self.cs_method == 'cspf':
                self.steps.append(('spatial_filter', RSF(dim=self.nelec, method='cspf')))
            else:
                raise ValueError('Invalid channel selection method.')
               
        # time_window_data_augmentation
        twda_flag = False
        if self.aug_method is not None:
            if self.aug_method == 'time_window':  
                twda_flag = True
                self.compat_flag = False  # compatibility flag
                self.window_width = end_time - start_time if (end_time is not None) and (start_time is not None) else self.window_width 
                self.steps.append(('time_window_data_augmentation', 
                                   TimeWindowDataExpansion(fs=self.fs_new, 
                                                           window_width=self.window_width, 
                                                           window_step=self.window_step if self.window_step is not None else 0.5
                                                           )))
        
        # time window selection
        if self.start_time is not None and self.end_time is not None:       
            self.steps.append(('time_window_selector', TimeWindowSelector(fs=self.fs_new, 
                                                                          start_time=self.start_time, 
                                                                          end_time=self.end_time, 
                                                                          twda_flag=twda_flag
                                                                          )))
        
        # remove mean
        self.steps.append(('remove_mean', RemoveMean()))
        
        # initialize the pipeline
        self.process = Pipeline(steps=self.steps, memory=self.memory)
        
    def fit(self, X, y=None):
        """
        Fit the pre-processing pipeline.

        Args:
            X (ndarray): input data. Shape (n_samples, n_channels, n_times)
            y (ndarray or None): labels. Shape (n_samples,)  

        Returns:
            self (Pre_Processing): the fitted pre-processing pipeline.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        self.process.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform the input data using the pre-processing pipeline.

        Args:
            X (ndarray): input data. Shape (n_samples, n_channels, n_times)

        Returns:
            X (ndarray): transformed data. Shape (n_samples, n_channels, n_times)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        return self.process.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the pre-processing pipeline and transform the input data using the pre-processing pipeline.

        Args:
            X (ndarray): input data. Shape (n_samples, n_channels, n_times)
            y (ndarray or None): labels. Shape (n_samples,)  

    Returns:
        X (ndarray): transformed data. Shape (n_samples, n_channels, n_times)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        return self.process.fit_transform(X, y)
    