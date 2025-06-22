"""
This file contains the implementation of the algorithms used in the MI-All-dev project.
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/21
License: BSD 3-Clause License
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pre_processing.preprocessing import Pre_Processing
from transfer_learning.tl_classifier import TL_Classifier
from joblib import Memory
from .utilities import estimator_list
from .base import decode_domains

def check_sample_dims(X, y):
    """
    展开X和y，使X变成标准的三维样本（不同通道成分数目*不同时间窗成分数目*多个频带成分数目*样本数）*通道数*时间点数
    并且扩展y以匹配新的样本维度。

    Parameters:
    X (np.ndarray): 输入数据，维度为(不同通道成分, 不同时间窗成分, 多个频带成分, ..., 样本数, 通道数, 时间点数)
    y (np.ndarray): 标签数据，维度为(样本数, 1)

    Returns:
    tuple: (新的X, 新的y)
        - 新的X: 维度为(新的样本数, 通道数, 时间点数)
        - 新的y: 维度为(新的样本数, 1)
    """
    # 获取输入X的维度
    input_shape = X.shape
    
    # 检查输入X的维度是否正确
    if len(input_shape) < 3:
        raise ValueError("输入X的维度不正确，至少需要3维。")
    elif  len(input_shape) == 3:  # 输入X的维度为(样本数, 通道数, 时间点数)
        return X, y

    # 样本数、通道数、时间点数
    sample_count, channel_count, time_point_count = input_shape[-3], input_shape[-2], input_shape[-1]

    # 计算新的样本数
    new_sample_count = np.prod(input_shape[:-3]) * sample_count

    # 重塑X
    new_X = X.reshape((new_sample_count, channel_count, time_point_count))

    # 扩展y
    new_y = np.repeat(y, np.prod(input_shape[:-3]), axis=0)

    return new_X, new_y

class Algorithms(BaseEstimator, ClassifierMixin):
    """
    This class contains the implementation of the algorithms used in the MI-All-dev project.
    
    Parameters
    ----------  
    algorithm_id: list
        A list containing the algorithm IDs for DPA, FEE, FES, CLF, END, and END-to-END.
    target_domain: str, optional
        The target domain for transfer learning.
    memory_location: str, optional
        The location of the cache memory.
    fs_new: int, optional
        The sampling frequency of the new data.
    fs_old: int, optional
        The sampling frequency of the old data.
    n_channels: int, optional
        The number of channels in the data.
    start_time: float, optional
        The start time of the data. 
    end_time: float, optional
        The end time of the data.
    lowcut: float, optional
        The low cut-off frequency of the filter.
    highcut: float, optional
        The high cut-off frequency of the filter. 
    aug_method: str, optional
        The augmentation method used for data augmentation.
    window_width: float, optional
        The width of the window for feature extraction.
    window_step: float, optional
        The step of the window for feature extraction.
    pre_est: str, optional
        The pre-trained estimator used for feature extraction.
    tl_mode: str, optional
        The transfer learning mode. The default is 'TL'.
    **kwargs: dict, optional
        The additional parameters for the algorithms.
    
    Attributes
    ----------
    DPA_METHODS: list
        A list containing the DPA methods.
    FEE_METHODS: list
        A list containing the FEE methods.
    FES_METHODS: list
        A list containing the FES methods.
    CLF_METHODS: list
        A list containing the CLF methods.
    END_METHODS: list
        A list containing the END methods.
    END_TO_END_METHODS: list
        A list containing the END-to-END methods.
    PreProcess: Pre_Processing
        An instance of the Pre_Processing class.
    TLClassifierModel: TL_Classifier
        An instance of the TL_Classifier class.
    Model: BaseEstimator
        The trained algorithm.
    
    Methods
    -------
    fit(X, y):
        This function trains the algorithm on the given data.
    predict(X):
        This function predicts the target data based on the input data.
    predict_proba(X):
        This function predicts the probability of the target data based on the input data.
    score(X, y):
        This function returns the accuracy of the algorithm on the given data.
    """
    def __init__(
        self, 
        algorithm_id, 
        target_domain=None, 
        *,
        memory_location=None,
        fs_new=None, 
        fs_old=None, 
        channels=None, 
        start_time=None, 
        end_time=None, 
        lowcut=None, 
        highcut=None, 
        aug_method=None,
        window_width=None,
        window_step=None,
        pre_est=None,
        tl_mode='TL',
        **kwargs
        ): 
        
        self.algorithm_id = algorithm_id
        self.target_domain = target_domain
        self.memory_location = memory_location
        self.fs_new = fs_new
        self.fs_old = fs_old
        self.channels = channels
        self.start_time = start_time
        self.end_time = end_time
        self.lowcut = lowcut
        self.highcut = highcut
        self.aug_method = aug_method
        self.window_width = window_width
        self.window_step = window_step
        self.pre_est = pre_est
        self.tl_mode = tl_mode  
        self.kwargs = kwargs

        if self.memory_location is not None:
            self.memory = Memory(location=self.memory_location, verbose=0, bytes_limit=1024*1024*1024*20)
        else:
            self.memory = None
        
        (self.DPA_METHODS, 
         self.FEE_METHODS, 
         self.FES_METHODS, 
         self.CLF_METHODS, 
         self.END_METHODS, 
         self.END_TO_END_METHODS) = estimator_list()
        
        # 实例化预处理类
        self.PreProcess = Pre_Processing(
            fs_new=self.fs_new, 
            fs_old=self.fs_old, 
            channels=self.channels, 
            start_time=self.start_time, 
            end_time=self.end_time, 
            lowcut=self.lowcut, 
            highcut=self.highcut, 
            aug_method=self.aug_method,
            window_width=self.window_width,
            window_step=self.window_step,
            memory=self.memory,
            **self.kwargs
            )
        
        self.TLClassifierModel = TL_Classifier(
            dpa_method=self.DPA_METHODS[self.algorithm_id[0]], 
            fee_method=self.FEE_METHODS[self.algorithm_id[1]], 
            fes_method=self.FES_METHODS[self.algorithm_id[2]], 
            clf_method=self.CLF_METHODS[self.algorithm_id[3]], 
            end_method=self.END_METHODS[self.algorithm_id[4]], 
            ete_method=self.END_TO_END_METHODS[self.algorithm_id[5]], 
            pre_est=self.pre_est, 
            memory=self.memory, 
            target_domain=self.target_domain,
            tl_mode=self.tl_mode,
            **self.kwargs
            )
        
    def fit(self, X, y):
        """
        This function trains the algorithm on the given data.
        :param X: The input data.
        :param y: The target data.
        :return: The trained algorithm.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X_dec, _, domains = decode_domains(X, y)
        w = np.zeros(len(X_dec))
        w[domains == self.target_domain] = 1
        for name, step in self.PreProcess.process.steps:
            if name == 'channel_selector_plus':
                step.weights = w
                break 
        
        if self.PreProcess.compat_flag:
            self.TLClassifierModel.__class__(
                dpa_method=self.DPA_METHODS[self.algorithm_id[0]], 
                fee_method=self.FEE_METHODS[self.algorithm_id[1]], 
                fes_method=self.FES_METHODS[self.algorithm_id[2]], 
                clf_method=self.CLF_METHODS[self.algorithm_id[3]], 
                end_method=self.END_METHODS[self.algorithm_id[4]], 
                ete_method=self.END_TO_END_METHODS[self.algorithm_id[5]], 
                pre_est=self.PreProcess.process, 
                memory=self.memory, 
                target_domain=self.target_domain,
                tl_mode=self.tl_mode,
                **self.kwargs
                )
            self.Model = self.TLClassifierModel.fit(X, y)
        else:
            X = self.PreProcess.fit_transform(X, y)
            X, y = check_sample_dims(X, y)  
            self.Model = self.TLClassifierModel.fit(X, y)
        return self
          
    def predict(self, X):
        """
        This function predicts the target data based on the input data.
        :param X: The input data.
        :return: The predicted target data.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if not self.PreProcess.compat_flag:
            X = self.PreProcess.transform(X)
        return self.Model.predict(X)

    def predict_proba(self, X):
        """
        This function predicts the probability of the target data based on the input data.
        :param X: The input data.
        :return: The predicted probability of the target data.
        """ 
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if not self.PreProcess.compat_flag:
            X = self.PreProcess.transform(X)
        return self.Model.predict_proba(X)

    def score(self, X, y):
        """
        This function returns the accuracy of the algorithm on the given data.
        :param X: The input data.
        :param y: The target data.
        :return: The accuracy of the algorithm.
        """ 
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if not self.PreProcess.compat_flag:
            X = self.PreProcess.transform(X)
        return self.Model.score(X, y)


            
            



