# original code: from pyriemann.spatialfilters import CSP
# modified by LC.Pan on 2024/6/23
# add sample_weight parameter to compute class means with weighted covariance matrices

import numpy as np
from numpy import ndarray
from scipy.linalg import eigh
from scipy import linalg
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.ajd import ajd_pham
from pyriemann.utils.utils import check_weights
from pyriemann.spatialfilters import CSP as CSP_Cov
from moabb.pipelines.csp import TRCSP
from metabci.brainda.algorithms.decomposition.csp import CSP

from ..utils import generate_intervals, adjust_intervals
from .cca import FilterBank
from .base import generate_filterbank


class CSP_weighted(CSP_Cov):
    def fit(self, X, y, sample_weight=None):
        """Train CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : CSP instance
            The CSP instance.
        """
        if not isinstance(self.nfilter, int):
            raise TypeError('nfilter must be an integer')
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_channels')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')
        
        sample_weight = check_weights(sample_weight, len(y))

        classes = np.unique(y)
        
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(X[y == c], self.metric, sample_weight=sample_weight[y == c]))
        C = np.array(C)

        # Switch between binary and multiclass
        if len(classes) == 2:
            evals, evecs = eigh(C[1], C[0] + C[1])
            # sort eigenvectors
            ix = np.argsort(np.abs(evals - 0.5))[::-1]
        elif len(classes) > 2:
            evecs, D = ajd_pham(C)
            sample_weight_ = np.array([sample_weight[y == c].sum() for c in classes])
            Ctot = mean_covariance(C, self.metric, sample_weight=sample_weight_)    
            evecs = evecs.T

            # normalize
            for i in range(evecs.shape[1]):
                tmp = evecs[:, i].T @ Ctot @ evecs[:, i]
                evecs[:, i] /= np.sqrt(tmp)

            mutual_info = []
            # class probability
            Pc = [np.mean(y == c) for c in classes]
            for j in range(evecs.shape[1]):
                a = 0
                b = 0
                for i, c in enumerate(classes):
                    tmp = evecs[:, j].T @ C[i] @ evecs[:, j]
                    a += Pc[i] * np.log(np.sqrt(tmp))
                    b += Pc[i] * (tmp ** 2 - 1)
                mi = - (a + (3.0 / 16) * (b ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]
        else:
            raise ValueError("Number of classes must be >= 2.")
        
        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        A = np.linalg.pinv(evecs.T)

        self.filters_ = evecs[:, 0:self.nfilter].T
        self.patterns_ = A[:, 0:self.nfilter].T

        return self

class TRCSP_weighted(TRCSP):
    def fit(self, X, y, sample_weight=None):
        """Train spatial filters.   
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : TRCSP instance
            The TRCSP instance.
        
        Only deals with two class
        """

        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X must be an array.")
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError("y must be an array.")
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError("X must be n_trials * n_channels * n_channels")
        if len(y) != len(X):
            raise ValueError("X and y must have the same length.")
        if np.squeeze(y).ndim != 1:
            raise ValueError("y must be of shape (n_trials,).")
        
        sample_weight = check_weights(sample_weight, len(y))

        n_trials, n_channels, _ = X.shape
        classes = np.unique(y)
        assert len(classes) == 2, "Can only do 2-class TRCSP"
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(X[y == c], self.metric, sample_weight=sample_weight[y == c]))
        C = np.array(C)

        # regularize CSP
        evals = [[], []]
        evecs = [[], []]
        Creg = C[1] + np.eye(C[1].shape[0]) * self.alpha
        evals[1], evecs[1] = linalg.eigh(C[0], Creg)
        Creg = C[0] + np.eye(C[0].shape[0]) * self.alpha
        evals[0], evecs[0] = linalg.eigh(C[1], Creg)
        # sort eigenvectors
        filters = []
        patterns = []
        for i in range(2):
            ix = np.argsort(evals[i])[::-1]  # in descending order
            # sort eigenvectors
            evecs[i] = evecs[i][:, ix]
            # spatial patterns
            A = np.linalg.pinv(evecs[i].T)
            filters.append(evecs[i][:, : (self.nfilter // 2)])
            patterns.append(A[:, : (self.nfilter // 2)])
        self.filters_ = np.concatenate(filters, axis=1).T
        self.patterns_ = np.concatenate(patterns, axis=1).T

        return self

class FBCSP(FilterBank):
    """
    Filter-Bank CSP (FBCSP) algorithm.
    """
    def __init__(self, fs, nfilter=4, banks=None, n_components_select=None):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency.
        nfilter : int
            Number of CSP filters per bank to extract.
        bank : list of tuples
            List of filterbank intervals. default [(4,8), (8,12), ..., (36,40)].
        n_components_select : int
            Number of components to select using mutual information. If None,
            automatically select the number of components based on the cross-validation.
        """
        
        self.fs=fs
        self.nfilter=nfilter        
        self.n_components_select=n_components_select
        
        self.banks=generate_intervals(4, 4, (4, 40)) if banks is None else banks
        self.filterbanks = generate_filterbank(self.banks,
                                          adjust_intervals(self.banks),
                                          srate=self.fs,
                                          order=4)
        
        super().__init__(
            CSP(n_components=self.nfilter),
            filterbank=self.filterbanks,
            )
    
    def __repr__(self):
        return f"FBCSP(fs={self.fs}, nfilter={self.nfilter}, banks={self.banks}, " \
               f"n_components_select={self.n_components_select})"
    
    def fit(self, X: ndarray, y: ndarray):  # type: ignore[override]
        """
        Fit the filterbank CSP algorithm.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : FBCSP instance
            The FBCSP instance.
        """
        super().fit(X, y)
        features = super().transform(X)
        if self.n_components_select is None:
            estimator = make_pipeline(
                *[SelectKBest(score_func=mutual_info_classif, k="all"), SVC()]
            )
            params = {"selectkbest__k": np.arange(1, features.shape[1] + 1)}

            n_splits = np.min(np.unique(y, return_counts=True)[1])
            n_splits = 5 if n_splits > 5 else n_splits

            gs = GridSearchCV(
                estimator,
                param_grid=params,
                scoring="accuracy",
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True),
                refit=False,
                n_jobs=-1,
                verbose=False,
            )
            gs.fit(features, y)
            self.n_components_select = gs.best_params_["selectkbest__k"]

        self.selector_ = SelectKBest(
            score_func=mutual_info_classif, k=self.n_components_select
        )
        self.selector_.fit(features, y)
        return self

    def transform(self, X: ndarray):  # type: ignore[override]
        features = super().transform(X)
        features = self.selector_.transform(features)
        return features