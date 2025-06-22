# Off-line modeling program/ Transfer learning
#
# Authors: Corey Lin <panlincong@tju.edu.cn.com>
# Date: 2023/07/08
# License: BSD 3-Clause License

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso
import numpy as np
from numpy import ndarray
import math

class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """
    基于互信息的特征选择器。

    这个类可以用来从数据中选择最有用的k个特征。
    它继承自sklearn库中的BaseEstimator和TransformerMixin类，
    可以与其他sklearn库中的模型和转换器一起使用。

    参数
    ----
    k : int, 可选（默认为-1）
        要选择的特征数量。如果k为-1，则选择30%的特征; 如果k为0，则选择所有特征，即不进行特征选择。

    属性
    ----
    mutual_infos_ : ndarray, shape (n_features,)
        每个特征与目标变量之间的互信息。

    方法
    ----
    fit(X, y) : 计算每个特征与目标变量之间的互信息，并选择互信息最大的k个特征。
    transform(X) : 选择数据中的特征。
    _get_support_mask() : 返回一个布尔掩码，表示哪些特征被选择。
    """
    def __init__(self, k:int = -1):
        self.k = k
    
    def _get_support_mask(self):
        """
        返回一个布尔掩码，表示哪些特征被选择。

        返回值
        ------
        mask : ndarray, shape (n_features,)
            布尔掩码，表示哪些特征被选择。
        """
        # 选择互信息最大的k个特征
        mask = np.zeros_like(self.mutual_infos_, dtype=bool)
        mask[np.argsort(self.mutual_infos_)[-self.k:]] = True
        return mask

    def fit(self, X: ndarray, y):
        """
        计算每个特征与目标变量之间的互信息，并选择互信息最大的k个特征。

        参数
        ----
        X : ndarray, shape (n_samples, n_features)
            训练数据的特征矩阵。
        y : ndarray, shape (n_samples,)
            训练数据的标签向量。

        返回值
        ------
        self : 返回拟合后的实例。
        """
        if self.k == -1:
            self.k = math.ceil(0.3 * X.shape[1])
        # 计算每个特征与目标变量之间的互信息
        self.mutual_infos_ = mutual_info_classif(X, y)
        return self
    
    def transform(self, X: ndarray):
        """
        选择数据中的特征。

        参数
        ----
        X : ndarray, shape (n_samples, n_features)
            要选择特征的数据矩阵。

        返回值
        ------
        X_new : ndarray, shape (n_samples, k)
            选择后的数据矩阵，其中只包含被选择的k个特征。
        """
        mask = self._get_support_mask()
        return X[:, mask]

    

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.lasso = Lasso(alpha=self.alpha)
        
    def fit(self, X, y=None):
        self.lasso.fit(X, y)
        self.support_ = np.where(self.lasso.coef_ != 0)[0]
        return self
    
    def transform(self, X):
        return X[:, self.support_]
