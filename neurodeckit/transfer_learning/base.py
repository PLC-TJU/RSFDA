# Authors: LC.Pan <panlincong@tju.edu.cn>
# Date: 2024/4/7
# License: BSD 3-Clause License

import numpy as np
from typing import List, Union, Optional
from numpy import ndarray

from sklearn.base import BaseEstimator, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from inspect import signature

def _combine_datasets(Xt: ndarray, yt: ndarray, Xs: Optional[Union[ndarray, List[ndarray]]] = None, ys: 
    Optional[Union[ndarray, List[ndarray]]] = None) -> (List[ndarray], List[ndarray]): # type: ignore
    """
    结合来自不同来源的数据集。

    参数:
    Xt (ndarray): 目标域的数据集，形状为 (样本数, 通道数, 采样点数)。
    yt (ndarray): 目标域的标签向量，长度为样本数。
    Xs (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的数据集，可以是以下之一:
        - None: 不使用源域数据。
        - 3维ndarray: 单个源域的数据集，形状为 (样本数, 通道数, 采样点数)。
        - 列表: 包含多个源域数据集的列表，每个元素都是一个3维ndarray。
        - 4维ndarray: 包含多个源域数据集的4维数组，形状为 (数据集数量, 样本数, 通道数, 采样点数)。
    ys (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的标签集，可以是以下之一:
        - None: 不使用源域标签。
        - 1维ndarray: 单个源域的标签向量，长度为样本数。
        - 列表: 包含多个源域标签向量的列表，每个元素都是一个1维ndarray。

    返回:
    (List[ndarray], List[ndarray]): 两个列表，第一个是数据集列表，第二个是对应的标签列表。每个列表的元素数量等于“数据集来源数”。

    示例:
    >>> Xt = np.random.rand(10, 5, 100)  # 目标域数据集
    >>> yt = np.random.randint(0, 2, 10)  # 目标域标签
    >>> Xs = [np.random.rand(8, 5, 100), np.random.rand(12, 5, 100)]  # 源域数据集列表
    >>> ys = [np.random.randint(0, 2, 8), np.random.randint(0, 2, 12)]  # 源域标签列表
    >>> X_combined, Y_combined = combine_datasets(Xt, yt, Xs, ys)
    """

    X_combined = [Xt]  # 初始化数据集列表，首先添加目标域数据集
    Y_combined = [yt]  # 初始化标签列表，首先添加目标域标签

    # 检查Xs是否为None
    if Xs is None:
        return X_combined, Y_combined

    # 检查Xs是否为列表
    if isinstance(Xs, list):
        # 确保ys也是列表且长度与Xs相同
        if not isinstance(ys, list) or len(ys) != len(Xs):
            raise ValueError("ys must be a list with the same length as Xs")
        X_combined.extend(Xs)
        Y_combined.extend(ys)

    # 检查Xs是否为4维数组
    elif Xs.ndim == 4:
        # 将4维数组拆分为多个3维数组并添加到列表中
        for i in range(Xs.shape[0]):
            X_combined.append(Xs[i])
            Y_combined.append(ys[i])

    # 检查Xs是否为3维数组
    elif Xs.ndim == 3:
        X_combined.append(Xs)
        Y_combined.append(ys)

    # 其他情况，抛出异常
    else:
        raise ValueError("Xs must be either a 3D or 4D ndarray or a list of 3D ndarrays")

    return X_combined, Y_combined

# 与decode_domains搭配使用: from pyriemann.transfer import decode_domains
def combine_and_encode_datasets(Xt: ndarray, yt: ndarray, Xs: Optional[Union[ndarray, List[ndarray]]] = None, ys:
    Optional[Union[ndarray, List[ndarray]]] = None, Tags: Optional[List[str]] = None) -> (ndarray, ndarray): # type: ignore
    """
    组合来自不同源域的数据集，并对标签进行编码以区分不同的源域。

    参数:
    Xt (ndarray): 目标域的数据集，形状为 (样本数, 通道数, 采样点数)。
    yt (ndarray): 目标域的标签向量，长度为样本数。
    Xs (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的数据集。
    ys (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的标签集。
    Tags (Optional[List[str]], 默认为None): 定义不同来源数据集的标记列表。

    返回:
    (ndarray, ndarray): 所有数据集来源的数据集的组合的三维数组X，以及带有不同数据集domain标记的标签。
    """
    # 如果Tags为空，则生成默认的Tags列表
    if Tags is None:
        num_datasets = 1 if Xs is None else (len(Xs) + 1 if isinstance(Xs, list) or Xs.ndim == 4 else 2)
        Tags = ['S' + str(i+1) for i in range(num_datasets)]

    # 调用combine_datasets函数合并数据集
    X_combined, Y_combined = _combine_datasets(Xt, yt, Xs, ys)

    # 将合并后的数据集转换为三维数组
    X = np.concatenate(X_combined, axis=0)
    
    # 创建一个空列表来存储编码后的标签
    y_enc = []
    
    # 对每个源域的数据集进行编码
    for i, (X_domain, y_domain) in enumerate(zip(X_combined, Y_combined)):
        domain_tag = Tags[i]
        y_enc.extend([domain_tag + '/' + str(label) for label in y_domain])
    
    return X, np.array(y_enc)

# 与decode_domains搭配使用: from pyriemann.transfer import decode_domains
def encode_datasets(X: Union[List[ndarray], ndarray], 
                    Y: Union[List[ndarray], ndarray], 
                    domain_tags: Optional[List[str]] = None) -> (ndarray, ndarray, List[str]): # type: ignore
    """
    将来自不同源域的数据集整理并编码标签。

    参数:
    X (Union[List[ndarray], ndarray]): 不同来源的数据集，可以是列表或3/4维数组。
    Y (Union[List[ndarray], ndarray]): 相应的不同来源的标签集，可以是列表或1/2维数组。
    domain_tags (Optional[List[str]], 默认为None): 各个来源数据集的标记，长度应与数据集数量一致。

    返回:
    (ndarray, ndarray, List[str]): 整理后的数据集X和编码后的标签y_enc及domain_tags。
    """
    
    # 如果domain_tags为空，则生成默认的domain_tags列表
    if domain_tags is None:
        num_datasets = len(X) if isinstance(X, list) or X.ndim == 4 else 1
        domain_tags = ['S' + str(i+1) for i in range(num_datasets)]

    # 初始化编码后的数据集和标签列表
    X_encoded = []
    y_encoded = []

    # 检查X是否为列表
    if isinstance(X, list):
        # 确保Y也是列表且长度与X相同
        if not isinstance(Y, list) or len(Y) != len(X):
            raise ValueError("Y must be a list with the same length as X")
        for i, (X_i, Y_i) in enumerate(zip(X, Y)):
            X_encoded.append(X_i)
            y_encoded.extend([domain_tags[i] + '/' + str(y) for y in Y_i])
    
    # 检查X是否为4维数组
    elif X.ndim == 4:
        for i in range(X.shape[0]):
            X_encoded.append(X[i])
            y_encoded.extend([domain_tags[i] + '/' + str(y) for y in Y[i]])
    
    # 检查X是否为3维数组
    elif X.ndim == 3:
        return X, Y, domain_tags

    # 其他情况，抛出异常
    else:
        raise ValueError("X must be either a list of 3D ndarrays, a 4D ndarray, or a single 3D ndarray")

    # 将编码后的数据集转换为三维数组
    X_encoded = np.concatenate(X_encoded, axis=0)
    
    return X_encoded, np.array(y_encoded), domain_tags

#引用自 pyriemann.transfer import decode_domains
def decode_domains(X_enc, y_enc):
    """Decode the domains of the matrices in the labels.

    We handle the possibility of having different domains for the datasets by
    encoding the domain information into the labels of the matrices. This
    method converts the data into its original form, with a separate data
    structure for labels and for domains.

    Parameters
    ----------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each matrix.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Domains for each matrix.

    See Also
    --------
    encode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    y, domain = [], []
    if '/' not in str(y_enc[0]):
        return X_enc, y_enc, np.array(['S1' for _ in range(len(y_enc))])
    
    for y_enc_ in y_enc:
        y_dec_ = y_enc_.split('/')
        domain.append(y_dec_[-2])
        y.append(y_dec_[-1])
        y = [int(i) for i in y]
    return X_enc, np.array(y), np.array(domain)



class TLSplitter:
    """Class for handling the cross-validation splits of multi-domain data.

    This is a wrapper to sklearn's cross-validation iterators [1]_ which
    ensures the handling of domain information with the data points. In fact,
    the data from source domain is always fully available in the training
    partition whereas the random splits are done on the data points from the
    target domain.

    Parameters
    ----------
    target_domain : str
        Domain considered as target.
    cv : float | BaseCrossValidator | BaseShuffleSplit, default=None
        An instance of a cross validation iterator from sklearn.
        if float, it is the fraction of the target domain data to use as the training set.
        if BaseCrossValidator or BaseShuffleSplit, it is used as the cross-validation iterator.
    no_calibration : bool, default=False
        Whether to use the entire target domain data as the test set.
        if True, the entire target domain is used as the test set (i.e. 
        calibration-free), otherwise a random split is done on the target 
        domain data.
    modeling : bool, default=False  
        if True, the whole source and target domain data is used for training,
        and the target domain data is used as the test set.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators

    Notes
    -----
    .. modified:: LC.Pan 2024/6/23
    """

    def __init__(self, target_domain, cv, no_calibration=False, modeling=False):
        self.target_domain = target_domain
        self.cv = cv 
        self.no_calibration = no_calibration
        self.modeling = modeling

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals.
        y : ndarray, shape (n_samples,)
            Extended labels for each sample.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # Decode the domains of the data points
        X, y, domain = decode_domains(X, y)

        # Identify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]
        y_target = y[idx_target]
        
        if self.modeling:
            # If modeling, we use all source and target domain data as the training 
            # set and the entire target domain as the test set
            train_idx = np.concatenate([idx_source, idx_target])
            test_idx = idx_target
            yield train_idx, test_idx
            return

        if self.no_calibration or self.cv == 0:
            # Use all target domain samples as the test set
            train_idx = idx_source
            test_idx = idx_target
            self.no_calibration = True
            self.cv = 0
            yield train_idx, test_idx
            return

        if isinstance(self.cv, float) and 0 < self.cv < 1:
            # the value of cv is a fraction of the target domain data to use as the training set
            train_idx = np.concatenate([idx_source, idx_target[:int(self.cv*len(idx_target))]])
            test_idx = idx_target[int(self.cv*len(idx_target)):]
            yield train_idx, test_idx
            return

        else:
            # Index of training-split for the target data points
            ss_target = self.cv.split(idx_target, y_target, groups=groups)
            for train_sub_idx_target, test_sub_idx_target in ss_target:
                train_idx = np.concatenate(
                    [idx_source, idx_target[train_sub_idx_target]])
                test_idx = idx_target[test_sub_idx_target]
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if self.no_calibration or self.modeling:
            return 1
        else:
            return self.cv.get_n_splits(X, y)

def chesk_sample_weight(clf):
    if isinstance(clf, Pipeline):
        return chesk_sample_weight(clf.steps[-1][1])
    else:
        fit_method = getattr(clf, 'fit')
        params = signature(fit_method).parameters
        return 'sample_weight' in params

class TLClassifier(BaseEstimator):
    """Transfer learning wrapper for classifiers.

    This is a wrapper for any classifier that converts extended labels used in
    Transfer Learning into the usual y array to train a classifier of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseClassifier
        The classifier to apply on matrices.
    tl_mode : str, default='tl'
        The transfer learning model to use.
        'TL' : Transfer Learning (default) - train the classifier on the source and target domain data.
        'NOTL' : No Transfer Learning, i.e. train the classifier on the target domain data only.
        'CALIBRATION-FREE' : Calibration-Free Transfer Learning, i.e. train the classifier on the source domain data only.
    domain_weight : None | dict, default=None
        Weights to combine matrices from each domain to train the classifier.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    Notes
    -----
    .. modified:: LC.Pan 2024/6/23
    """
    
    def __init__(self, target_domain, estimator, tl_mode='tl', domain_weight=None):
        """Init."""
        self.target_domain = target_domain
        self.tl_mode = tl_mode
        self.domain_weight = domain_weight
        self.estimator = estimator
        
        if not is_classifier(self.estimator):
            raise TypeError('Estimator has to be a classifier.')

    def fit(self, X, y_enc):
        """Fit TLClassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLClassifier instance
            The TLClassifier instance.
        """

        X_dec, y_dec, domains = decode_domains(X, y_enc)

        if self.tl_mode.upper() == 'TL':
            if self.domain_weight is not None:
                w = np.zeros(len(X_dec))
                for d in np.unique(domains):
                    w[domains == d] = self.domain_weight[d]
            else:
                w = None
        elif self.tl_mode.upper() in ['NOTL', 'NT']:
            w = np.zeros(len(X_dec))
            w[domains == self.target_domain] = 1
        elif self.tl_mode.upper() in ['CALIBRATION-FREE','CF']:
            w = np.ones(len(X_dec))
            if self.domain_weight is not None:
                for d in np.unique(domains):
                    w[domains == d] = self.domain_weight[d] 
            w[domains == self.target_domain] = 0
        else:
            raise ValueError('tl_model should be either "TL", "NOTL" or "CALIBRATION-FREE".')

        # check if there are samples to train the model
        if w is not None and w.size == 0:
            raise ValueError('No samples to train the model.')
        
        # Excluding samples with a weight of 0
        X_dec = X_dec[w != 0] if w is not None else X_dec
        y_dec = y_dec[w != 0] if w is not None else y_dec
        w = w[w != 0] if w is not None else w
        
        # Fit the estimator
        if isinstance(self.estimator, Pipeline):
            sample_weight = {}
            for step in self.estimator.steps:
                step_name = step[0]
                step_pipe = step[1]
                if chesk_sample_weight(step_pipe):
                    sample_weight[step_name + '__sample_weight'] = w

            self.estimator.fit(X_dec, y_dec, **sample_weight)
        else:
            if chesk_sample_weight(self.estimator):         
                self.estimator.fit(X_dec, y_dec, sample_weight=w)
            else:
                self.estimator.fit(X_dec, y_dec)

        return self
    
    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each matrix according to the estimator.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices, n_classes)
            Predictions for each matrix.
        """
        return self.estimator.predict_proba(X)

    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Test set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended true labels for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)