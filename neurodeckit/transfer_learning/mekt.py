""" 
modified from metabci.brainda.algorithms.transfer_learning.mekt.py
by LC.Pan at 2024-06-23
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import OneHotEncoder
from scipy.linalg import block_diag, eigh
from scipy.spatial.distance import pdist, squareform
from pyriemann.utils import mean_covariance
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.utils import check_weights
from . import decode_domains

def source_discriminability(Xs, ys):
    """Source features discriminability.

    Parameters
    ----------
    Xs : ndarray
        source features, shape (n_trials, n_features)
    ys : ndarray
        labels, shape (n_trials)

    Returns
    -------
    Sw: ndarray
        within-class scatter matrix, shape (n_features, n_features)
    Sb: ndarray
        between-class scatter matrix, shape (n_features, n_features)
    """
    classes = np.unique(ys)
    n_samples, n_features = Xs.shape
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    mean_total = np.mean(Xs, axis=0, keepdims=True)

    for c in classes:
        Xi = Xs[ys == c, :]
        mean_class = np.mean(Xi, axis=0, keepdims=True)
        Sw = Sw + (Xi - mean_class).T @ (Xi - mean_class)
        Sb = Sb + (mean_class.T - mean_total.T) @ (mean_class - mean_total) * len(Xi)

    return Sw, Sb


def graph_laplacian(Xs, k=10, t=1):
    """Graph Laplacian Matrix.

    Currently with heat kernel implemented.

    Parameters
    ----------
    Xs : ndarray
        features, shape (n_trials, n_samples)
    k : int, optional
        k nearest neighbors, by default 10
    t : int, optional
        heat kernel parameter, by default 1

    Returns
    -------
    L: ndarray
        unnormalized laplacian kernel, shape (n_trials, n_trials)
    D: ndarray
        degree matrix, L = D - W, shape (n_trials, n_trials)
    """

    # compute pairwise distance
    pair_dist = squareform(pdist(Xs, metric="euclidean"))

    # knn
    # MEKT has self-connection, W[0,0] = 1
    ix = np.argsort(pair_dist, axis=-1)[:, : k + 1]

    # heat kernel
    heat_W = np.exp(-np.square(pair_dist) / (2 * np.square(t)))
    W = np.zeros((Xs.shape[0], Xs.shape[0]))

    for i, ind in enumerate(ix):
        W[i, ind] = heat_W[i, ind]

    W = np.maximum(W, W.T)

    D = np.diag(np.sum(W, axis=-1))
    L = D - W

    return L, D


def mekt_feature(X, sample_weight=None, metric='riemann'):
    """Covariance Matrix Centroid Alignment and Tangent Space Feature Extraction.
       Parameters
    ----------
    X : ndarray
        EEG data, shape (n_trials, n_channels, n_channels)

    Returns
    -------
    feature: ndarray
        feature of X, shape (n_trials, n_feature)

    """

    # Covariance Matrix Centroid Alignment
    M = mean_covariance(X, metric=metric, sample_weight=sample_weight)
    iM12 = invsqrtm(M)
    C = iM12 @ X @ iM12.T
    # Tangent Space Feature Extraction
    feature = tangent_space(C, np.eye(M.shape[0]), metric=metric)

    return feature

def mekt_kernel(Xs, Xt, ys, yt=None, d=10, max_iter=5, alpha=0.01, beta=0.1, rho=20, k=10, t=1, clf=None):
    """Manifold Embedding Knowledge Transfer.

    Parameters
    ----------
    Xs : ndarray
        source features, shape (n_source_trials, n_features)
    Xt : ndarray
        target features, shape (n_target_trials, n_features)
    ys : ndarray
        source labels, shape (n_source_trials,)
    yt : ndarray, optional
        target labels, shape (n_target_trials,), by default None
    d : int, optional
        selected d projection vectors, by default 10
    max_iter : int, optional
        max iterations, by default 5
    alpha : float, optional
        regularized term for source domain discriminability, by default 0.01
    beta : float, optional
        regularized term for target domain locality, by default 0.1
    rho : int, optional
        regularized term for parameter transfer, by default 20
    k : int, optional
        number of nearest neighbors
    t : int, optional
        heat kernel parameter
    clf : object, optional
        classifier for source domain discriminability, by default LDA(solver='lsqr', shrinkage='auto')

    Returns
    -------
    A: ndarray
        projection matrix for source, shape (n_features, d)
    B: ndarray
        projection matrix for target, shape (n_features, d)
    """
    ns_samples, ns_features = Xs.shape
    nt_samples, nt_features = Xt.shape

    # source domain discriminability
    Sw, Sb = source_discriminability(Xs, ys)
    P = np.zeros((2 * ns_features, 2 * ns_features))
    P[:ns_features, :ns_features] = Sw
    P0 = np.zeros((2 * ns_features, 2 * ns_features))
    P0[:ns_features, :ns_features] = Sb

    # target locality
    L, D = graph_laplacian(Xt, k=k, t=t)  # should be (n_samples, n_samples)
    iD12 = invsqrtm(D)
    L = iD12 @ L @ iD12
    L = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ L @ Xt)

    Q = np.block(
        [
            [np.eye(ns_features), -1 * np.eye(nt_features)],
            [-1 * np.eye(ns_features), 2 * np.eye(nt_features)],
        ]
    )

    Ht = np.eye(nt_samples) - (1 / nt_samples) * np.ones((nt_samples, 1)) @ np.ones(
        (1, nt_samples)
    )
    S = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ Ht @ Xt)

    classes = np.sort(np.unique(ys))
    onehot_enc = OneHotEncoder(categories=[classes], sparse_output=False)
    Ns = onehot_enc.fit_transform(np.reshape(ys, (-1, 1))) / len(ys)

    if yt is None:
        # 原本（来自metabci.brainda.algorithms.transfer_learning.mekt.py）
        # 存在问题：这一步实际不太符合无监督，因为现实中无法在初始获取所有的测试样本
        clf = LDA(solver="lsqr", shrinkage="auto") if clf is None else clf
        yt = clf.fit(Xs, ys).predict(Xt)  # initial predict label 
    else:
        # 有监督版本 （LC.Pan 2024-09-01 修改）
        max_iter = 1

    X = block_diag(Xs, Xt)
    Emin_temp = alpha * P + beta * L + rho * Q
    Emax = S + alpha * P0 + 1e-3 * np.eye(ns_features + nt_features)
    for _ in range(max_iter):
        # update fake yt
        Nt = onehot_enc.fit_transform(np.reshape(yt, (-1, 1))) / len(yt)

        # calculate R: joint probability distribution shift
        M = np.block([[Ns @ Ns.T, -Ns @ Nt.T], [-Nt @ Ns.T, Nt @ Nt.T]])
        R = X.T @ M @ X

        # generalized eigen-decompostion
        Emin = Emin_temp + R

        w, V = eigh(Emin, Emax)

        A = V[:ns_features, :d]
        B = V[ns_features:, :d]

        # embedding
        Zs = Xs @ A
        Zt = Xt @ B

        yt = clf.fit(Zs, ys).predict(Zt)

    return A, B

class MEKT(BaseEstimator, ClassifierMixin): #基于特征空间的迁移学习方法，需要目标域样本，但不需要标签
    """
    Manifold Embedded Knowledge Transfer(MEKT) [1]_.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-01-22

    update log:
        2021-01-22 by Swolf <swolfforever@gmail.com>

        2023-12-09 by heoohuan <heoohuan@163.com>（Add code annotation）
        
        2024-06-23 by LC.Pan <LC.Pan@outlook.com>
         - add fit method
         - add transform method 
         - add predict method
         - add score method
         - add target_domain parameter to fit_transform method
         - add decode_domains function to decode source and target domains
         - add sample_weight parameter to mekt_feature method
         - add metric parameter to mekt_feature method
         - change X shape to (n_trials, n_channels, n_channels) in fit and fit_transform methods
         - add estimator parameter to MEKT class to make it compatible with sklearn API

    Parameters
    ----------
    subspace_dim: int
        Maximum number of selected projection vectors, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        Covariance category, by default 'lwf'.

    Attributes
    ----------
    subspace_dim: int
        Selected projection vector, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        covariance category, by default 'lwf'.
    A_: ndarray
        first type center, shape(n_class, n_channels, n_channels).
    B_: ndarray
       second type center, shape(n_class, n_channels, n_channels).

    Raises
    ----------
    ValueError
        None

    References
    ----------
    .. [1] Zhang W, Wu D. Manifold embedded knowledge transfer for brain-computer interfaces
       [J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28 (5): 1117–1127.

    """

    def __init__(
        self,
        target_domain,
        subspace_dim: int = 10,
        max_iter: int = 5,
        alpha: float = 0.01,
        beta: float = 0.1,
        rho: float = 20,
        k: int = 10,
        t: int = 1,
        metric="riemann",
        selector=None,
        estimator=LDA(solver='lsqr', shrinkage='auto'),
    ):
        self.target_domain = target_domain
        self.subspace_dim = subspace_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.k = k
        self.t = t
        self.metric = metric
        self.selector = selector
        self.estimator = estimator
        self.A_ = None
        self.B_ = None
        self.model_ = None

    def fit_transform_(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        
        featureXs = mekt_feature(
            Xs, 
            sample_weight=sample_weight[domains != self.target_domain], 
            metric=self.metric
            )
        featureXt = mekt_feature(
            Xt, 
            metric=self.metric
            )
        
        # 特征选择
        if self.selector is not None:
            featureXs_selected = self.selector.fit_transform(featureXs, ys)
            featureXt_selected = self.selector.transform(featureXt)
        else:
            featureXs_selected = featureXs
            featureXt_selected = featureXt
        
        self.A_, self.B_ = mekt_kernel(
            featureXs_selected,
            featureXt_selected,
            ys,
            yt=None,
            d=self.subspace_dim,
            max_iter=self.max_iter,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            k=self.k,
            t=self.t,
            clf=self.estimator,
        )
        source_features = featureXs_selected @ self.A_
        target_features = featureXt_selected @ self.B_
        # feature = np.concatenate((source_features, target_features), axis=0)
        feature = np.zeros((len(domains), source_features.shape[-1]))
        feature[domains != self.target_domain] = source_features
        feature[domains == self.target_domain] = target_features
        
        return source_features, ys # 只返回源域特征和标签
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the model with X and y.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        self: object
            Returns the instance itself. 

        """
        features, y = self.fit_transform_(X, y_enc, sample_weight)
        self.classes_ = np.unique(y)
        if self.estimator is not None:
            self.model_ = self.estimator.fit(features, y)
        return self
    
    def transform(self, X):
        """Obtain target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        target_features: ndarray
            target domain features, shape(n_trials, n_features).

        """
        feature = mekt_feature(X, metric=self.metric)
        
        if self.selector is not None:
            feature = self.selector.transform(feature)
        
        target_features = feature @ self.B_
        
        return target_features
    
    def predict(self, X):
        """Predict the target domain labels.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        y_pred: ndarray
            Predicted target domain labels, shape(n_trials,).

        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")
        
        y_pred = self.model_.predict(self.transform(X))

        return y_pred
    
    def predict_proba(self, X):
        """Predict the target domain probabilities.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        y_pred_proba: ndarray
            Predicted target domain probabilities, shape(n_trials, n_classes).

        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")

        y_pred_proba = self.model_.predict_proba(self.transform(X))

        return y_pred_proba
    
    def score(self, X, y_enc):
        """Calculate the accuracy of the model.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).

        Returns
        -------
        score: float
            Accuracy of the model.

        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

def mekt_kernel_improved(Xs, Xt, ys, yt, d=10, alpha_s=0.01, alpha_t=0.01, beta=0.1, rho=20, k=10, t=1):
    """Improved Manifold Embedding Knowledge Transfer.
    LC.Pan 2024-09-01

    Parameters
    ----------
    Xs : ndarray
        source features, shape (n_source_trials, n_features)
    Xt : ndarray
        target features, shape (n_target_trials, n_features)
    ys : ndarray
        source labels, shape (n_source_trials,)
    yt : ndarray, optional
        target labels, shape (n_target_trials,)
    d : int, optional
        selected d projection vectors, by default 10
    alpha_s : float, optional
        regularized term for source domain discriminability, by default 0.01
    alpha_t : float, optional
        regularized term for target domain discriminability, by default 0.01
    beta : float, optional
        regularized term for target domain locality, by default 0.1
    rho : int, optional
        regularized term for parameter transfer, by default 20
    k : int, optional
        number of nearest neighbors
    t : int, optional
        heat kernel parameter

    Returns
    -------
    A: ndarray
        projection matrix for source, shape (n_features, d)
    B: ndarray
        projection matrix for target, shape (n_features, d)
    """
    ns_samples, ns_features = Xs.shape
    nt_samples, nt_features = Xt.shape

    # source domain discriminability
    Sw, Sb = source_discriminability(Xs, ys)
    Ps = np.zeros((2 * ns_features, 2 * ns_features))
    Ps[:ns_features, :ns_features] = Sw
    Ps0 = np.zeros((2 * ns_features, 2 * ns_features))
    Ps0[:ns_features, :ns_features] = Sb
    
    # target domain discriminability
    Tw, Tb = source_discriminability(Xt, yt)
    Pt = np.zeros((2 * nt_features, 2 * nt_features))
    Pt[nt_features:, nt_features:] = Tw
    Pt0 = np.zeros((2 * nt_features, 2 * nt_features))
    Pt0[nt_features:, nt_features:] = Tb

    # target locality
    L, D = graph_laplacian(Xt, k=k, t=t)  # should be (n_samples, n_samples)
    iD12 = invsqrtm(D)
    L = iD12 @ L @ iD12
    L = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ L @ Xt)

    Q = np.block(
        [
            [np.eye(ns_features), -1 * np.eye(nt_features)],
            [-1 * np.eye(ns_features), 2 * np.eye(nt_features)],
        ]
    )

    Ht = np.eye(nt_samples) - (1 / nt_samples) * np.ones((nt_samples, 1)) @ np.ones(
        (1, nt_samples)
    )
    S = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ Ht @ Xt)

    classes = np.sort(np.unique(ys))
    onehot_enc = OneHotEncoder(categories=[classes], sparse_output=False)
    Ns = onehot_enc.fit_transform(np.reshape(ys, (-1, 1))) / len(ys)
    Nt = onehot_enc.fit_transform(np.reshape(yt, (-1, 1))) / len(yt)
    # calculate R: joint probability distribution shift
    X = block_diag(Xs, Xt)
    M = np.block([[Ns @ Ns.T, -Ns @ Nt.T], [-Nt @ Ns.T, Nt @ Nt.T]])
    R = X.T @ M @ X

    Emin = alpha_s * Ps + alpha_t * Pt + beta * L + rho * Q + R
    Emax = S + alpha_s * Ps0 + alpha_t * Pt0 + 1e-3 * np.eye(ns_features + nt_features)
    w, V = eigh(Emin, Emax)

    A = V[:ns_features, :d]
    B = V[ns_features:, :d]
    
    return A, B


# 纠正版，有监督学习的MEKT （LC.Pan 2024-09-01）
class MEKT_supervised(MEKT):
    def fit_transform_(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        yt = y[domains == self.target_domain]
        
        if Xt.size == 0:
            # raise ValueError("No target domain data found.")
            flag = False
        else:
            flag = True
        
        
        featureXs = mekt_feature(
            Xs, 
            sample_weight=sample_weight[domains != self.target_domain], 
            metric=self.metric
            )
        featureXt = mekt_feature(
            Xt, 
            metric=self.metric
            ) if flag else None
        
        # 特征选择
        if self.selector is not None:
            featureXs_selected = self.selector.fit_transform(featureXs, ys)
            featureXt_selected = self.selector.transform(featureXt) if flag else None
        else:
            featureXs_selected = featureXs
            featureXt_selected = featureXt if flag else None
        
        if not flag:
            self.B_ = np.eye(featureXs_selected.shape[1])
            return featureXs_selected, ys
        
        self.A_, self.B_ = mekt_kernel(
            featureXs_selected,
            featureXt_selected,
            ys,
            yt,
            d=self.subspace_dim,
            max_iter=self.max_iter,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            k=self.k,
            t=self.t,
            clf=self.estimator,
        )
        
        source_features = featureXs_selected @ self.A_
        target_features = featureXt_selected @ self.B_
        feature = np.zeros((len(domains), source_features.shape[-1]))
        feature[domains != self.target_domain] = source_features
        feature[domains == self.target_domain] = target_features
        
        return feature, y
    

# 改进版，有监督学习的MEKT （LC.Pan 2024-09-01）
class MEKT_improved(MEKT):
    def __init__(
        self,
        target_domain,
        subspace_dim: int = 10,
        alpha_s: float = 0.01,
        alpha_t: float = 0.01,
        beta: float = 0.1,
        rho: float = 20,
        k: int = 10,
        t: int = 1,
        metric="riemann",
        selector=None,
        estimator=LDA(solver='lsqr', shrinkage='auto'),
        **kwargs,
    ):
        self.target_domain = target_domain
        self.subspace_dim = subspace_dim
        self.alpha_s = alpha_s
        self.alpha_t = alpha_t
        self.beta = beta
        self.rho = rho
        self.k = k
        self.t = t
        self.metric = metric
        self.selector = selector
        self.estimator = estimator
        self.A_ = None
        self.B_ = None
        self.model_ = None
        
    def fit_transform_(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        yt = y[domains == self.target_domain]
        
        if Xt.size == 0:
            # raise ValueError("No target domain data found.")
            flag = False
        else:
            flag = True
          
        featureXs = mekt_feature(
            Xs, 
            sample_weight=sample_weight[domains != self.target_domain], 
            metric=self.metric
            )
        featureXt = mekt_feature(
            Xt, 
            metric=self.metric
            ) if flag else None
        
        # 特征选择
        if self.selector is not None:
            featureXs_selected = self.selector.fit_transform(featureXs, ys)
            featureXt_selected = self.selector.transform(featureXt) if flag else None
        else:
            featureXs_selected = featureXs
            featureXt_selected = featureXt if flag else None
        
        if not flag:
            self.B_ = np.eye(featureXs_selected.shape[1])
            return featureXs_selected, ys
        
        self.A_, self.B_ = mekt_kernel_improved(
            featureXs_selected,
            featureXt_selected,
            ys,
            yt,
            d=self.subspace_dim,
            alpha_s=self.alpha_s,
            alpha_t=self.alpha_t,
            beta=self.beta,
            rho=self.rho,
            k=self.k,
            t=self.t,
        )
        
        source_features = featureXs_selected @ self.A_
        target_features = featureXt_selected @ self.B_
        feature = np.zeros((len(domains), source_features.shape[-1]))
        feature[domains != self.target_domain] = source_features
        feature[domains == self.target_domain] = target_features
        
        return feature, y

# 区别于MEKT_improved，这里先进行MEKT再进行特征选择
class MEKT_improved2(MEKT_improved):
    def fit_transform_(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        yt = y[domains == self.target_domain]
        
        if Xt.size == 0:
            # raise ValueError("No target domain data found.")
            flag = False
        else:
            flag = True
        
        featureXs = mekt_feature(
            Xs, 
            sample_weight=sample_weight[domains != self.target_domain], 
            metric=self.metric
            )
        featureXt = mekt_feature(
            Xt, 
            metric=self.metric
            ) if flag else None
        
        if flag:
            self.A_, self.B_ = mekt_kernel_improved(
                featureXs,
                featureXt,
                ys,
                yt,
                d=self.subspace_dim,
                alpha_s=self.alpha_s,
                alpha_t=self.alpha_t,
                beta=self.beta,
                rho=self.rho,
                k=self.k,
                t=self.t,
            )
            
            featureXs = featureXs @ self.A_
            featureXt = featureXt @ self.B_
        
        # 特征选择
        if self.selector is not None:
            featureXs_selected = self.selector.fit_transform(featureXs, ys)
            featureXt_selected = self.selector.transform(featureXt) if flag else None
        else:
            featureXs_selected = featureXs
            featureXt_selected = featureXt if flag else None
        
        if not flag:
            self.B_ = np.eye(featureXs.shape[1])
            return featureXs_selected, ys
        
        feature = np.zeros((len(domains), featureXs_selected.shape[-1]))
        feature[domains != self.target_domain] = featureXs_selected
        feature[domains == self.target_domain] = featureXt_selected
        
        return feature, y
    
    def transform(self, X):
        """Obtain target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        target_features: ndarray
            target domain features, shape(n_trials, n_features).

        """
        feature = mekt_feature(X, metric=self.metric)
        feature = feature @ self.B_
        
        if self.selector is not None:
            feature = self.selector.transform(feature)
        
        return feature