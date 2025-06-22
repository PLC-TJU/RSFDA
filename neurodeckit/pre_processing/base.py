"""
This is the base.py file for preprocessing module.
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/21
License: BSD 3-Clause License
"""

import numpy as np
from scipy.signal import butter, cheby1, filtfilt, resample
from sklearn.base import BaseEstimator, TransformerMixin

# 带通滤波器类
class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, lowcut, highcut, order=5, filter_type='butter'):
        """
        带通滤波器类
        
        参数:
        lowcut (float): 带通滤波器的低频截止频率。
        highcut (float): 带通滤波器的高频截止频率。
        fs (float): 采样频率。
        order (int): 滤波器的阶数。
        filter_type (str): 滤波器类型，'butter' 或 'cheby1'。
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.filter_type = filter_type
    
    def __repr__(self): 
        return f"BandpassFilter(lowcut={self.lowcut}, highcut={self.highcut}, " \
               f"fs={self.fs}, order={self.order}, filter_type={self.filter_type})"
     
    def _get_filter_coeff(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        if self.filter_type == 'butter':
            self.b, self.a = butter(self.order, [low, high], btype='band')
        elif self.filter_type == 'cheby1':
            self.b, self.a = cheby1(self.order, 0.5, [low, high], btype='band')
        else:
            raise ValueError("filter_type must be 'butter' or 'cheby1'")    
        
        
    def _bandpass_filter(self, data):  

        return filtfilt(self.b, self.a, data, axis=-1)
        
    def fit(self, X, y=None, **fit_params): 
        self._get_filter_coeff()
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 带通滤波后的信号。
        """
        return self._bandpass_filter(X)

# 通道选择类
class ChannelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, channels):
        """
        通道选择类
        
        参数:
        channels (list): 选择的通道列表。
        """
        self.channels = channels
    
    def __repr__(self): 
        return f"ChannelSelector(channels={self.channels})"
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 选择的通道信号。
        """
        if len(X.shape) == 2:
            return X[self.channels]
        else:
            return X[:, self.channels, :]

# 时间窗选择类
class TimeWindowSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fs, start_time, end_time, **kwargs):
        """
        时间窗选择类
        
        参数:
        start (float): 起始时间。
        end (float): 终止时间。
        fs (float): 采样频率。
        
        可选参数:
        twda_flag (bool): 是否已经执行了滑动时间窗数据扩充。默认值为False。
        """
        self.start_time = start_time
        self.end_time = end_time
        self.fs = fs
        self.twda_flag = kwargs.get('twda_flag', False)
        
    def __repr__(self): 
        return f"TimeWindowSelector(start_time={self.start_time}, end_time={self.end_time}, fs={self.fs})"
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 选择的时间窗信号。
        """
        n_points = X.shape[-1]     
        t = np.arange(n_points) / self.fs
        idx = np.logical_and(t >= self.start_time, t < self.end_time)
        
        return X[..., idx]
    
    def fit_transform(self, X, y=None, **fit_params):     
        """  
        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() may be different than using .fit() and
           .transform() separately.
        
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 选择的时间窗信号。
        """
        return X if self.twda_flag else self.transform(X, y)
    
# 重采样类
class Downsample(BaseEstimator, TransformerMixin):
    def __init__(self, fs_old, fs_new): 
        """
        重采样类
        
        参数:
        fs_old (float): 原采样频率。
        fs_new (float): 目标采样频率。
        """
        self.fs_old = fs_old
        self.fs_new = fs_new
    
    def __repr__(self): 
        return f"Downsample(fs_old={self.fs_old}, fs_new={self.fs_new})"
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 重采样后的信号。
        """
        if self.fs_old == self.fs_new:
            return X
        
        n_points = int(X.shape[-1] * self.fs_new / self.fs_old)
        return resample(X, n_points, axis=-1)

# 去均值类
class RemoveMean(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        去均值类
        """
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 去均值后的信号。
        """
        X = X - np.mean(X, axis=-1, keepdims=True)
        return X

# 标准化类
class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        标准化类
        """
        self.mean = None
        self.std = None
    
    def fit(self, X, y=None, **fit_params):
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.std = np.std(X, axis=-1, keepdims=True)
        return self
    
    def transform(self, X, y=None):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 标准化后的信号。
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        """
        
        X = X - self.mean
        X = X / (self.std + 1e-10)
        
        return X

# 平滑类
class Smooth(BaseEstimator, TransformerMixin):
    def __init__(self, window_len, window='hanning'):
        """
        平滑类
        
        参数:
        window_len (int): 平滑窗口的长度。
        window (str): 平滑窗口类型。
        """
        self.window_len = window_len
        self.window = window
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        """  
        输入：
        X (array-like): 输入信号。 
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)
        
        输出：
        array-like: 平滑后的信号。
        """
        if self.window_len < 3:
            return X
        
        if self.window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        
        s = np.r_[X[0], X, X[-1]]
        if self.window == 'flat': # 平滑窗口为方波
            w = np.ones(self.window_len, 'd')
        else:
            w = eval('np.' + self.window + '(self.window_len)')
        
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[self.window_len:-self.window_len+1]

# 自定义转换器，将数据转换为 float32或float64 类型
class PrecisionConverter(BaseEstimator, TransformerMixin):
    def __init__(self, precision='float32'):
        self.precision = precision
        if self.precision not in ['float32', 'float64']:
            raise ValueError("precision must be 'float32' or 'float64'")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.astype(self.precision)

