"""
CTSSP: Common Temporal-Spectral-Spatial Patterns.
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/09/20
License: All rights reserved
"""

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.utils import check_weights
from pyriemann.utils.ajd import ajd_pham
from pyriemann.utils.base import invsqrtm, logm
from neurodeckit.machine_learning import sblest_kernel

def enhanced_cov_old(X, t_win, t_step, tau, whiten_filter=None, estimator='cov', metric='euclid', sample_weight=None):
    """
    Compute enhanced covariance matrices with whitening based on time window width, time step, and FIR filter time delays.

    Parameters:
    X (ndarray): Input EEG data of shape (n_trials, n_channels, n_timepoints).
    t_win (int): Time window width (in number of time points).
    t_step (int): Time window step (in number of time points).
    tau (int or list): Time delays (must be positive integers).
    estimator (str): Covariance estimator to use.
    metric (str): Distance metric to use.
    whiten_filter (ndarray): Whitening filter for each sub-window.
    sample_weight (ndarray): Sample weights.

    Returns:
    final_covariances: Enhanced whitened covariance matrices of shape (n_trials, n_channels * K * N, n_channels * K * N).
    whiten_filter: Whitening filter for each sub-window. shape (N, n_channels * K, n_channels * K).
    """
    num_samples, num_channels, num_time_points = X.shape
    is_train = True if whiten_filter is None else False
    if is_train:
        whiten_filter = []

    # Ensure tau is a list if it's a scalar
    if tau is None:  # If tau is not specified, use [0]
        tau = [0]
    if isinstance(tau, int):
        tau = [tau]
    K = len(tau)

    # 1. Calculate the number of time windows N
    if t_win > num_time_points:  # If t_win is larger than the signal length, use the whole signal as a single time window
        t_win = num_time_points
    N = (num_time_points - t_win) // t_step + 1

    # 2. Create sub-windows of the EEG signal
    sub_signals = []
    for i in range(N):
        start = i * t_step
        end = start + t_win
        sub_signal = X[:, :, start:end]  # Extract sub-window with shape (n_trials, n_channels, t_win)
        
        # 3. Apply time delays (tau) to each sub-window and concatenate along the channel dimension
        delayed_signals = []
        for delay in tau:
            if delay == 0:
                delayed_signal = sub_signal.copy()  # No shift needed
            else:
                delayed_signal = np.zeros_like(sub_signal)
                delayed_signal[:, :, delay:] = sub_signal[:, :, :-delay]  # Apply delay
                delayed_signal[:, :, :delay] = 0  # Zero padding for the first 'delay' points
            delayed_signals.append(delayed_signal)

        # Concatenate delayed signals along channel dimension for the current time window
        sub_signals.append(np.concatenate(delayed_signals, axis=1))  # Shape: (n_trials, n_channels * K, t_win)

    # 4. Compute covariance matrices and whiten each covariance matrix for each sub-window
    enhanced_covariances = []

    for i, sub_signal in enumerate(sub_signals):
        # Calculate covariance matrices for sub-signal
        cov_matrices = covariances(sub_signal, estimator=estimator)  # Shape: (n_trials, n_channels * K, n_channels * K)
        
        # Normalize whitened covariances by their trace
        cov_matrices /= np.maximum(
            np.trace(cov_matrices, axis1=1, axis2=2)[:, np.newaxis, np.newaxis], 
            1e-10
        )
        
        # Whitening step
        if is_train:
            mean_cov = mean_covariance(cov_matrices, metric=metric, sample_weight=sample_weight)
            whiten_filter.append(invsqrtm(mean_cov))# Calculate whitening filter for each sub-window
        whitened_cov_matrices = np.einsum('ij,bjk,kl->bil', whiten_filter[i].T, cov_matrices, whiten_filter[i])
        
        enhanced_covariances.append(whitened_cov_matrices)
        
    # 5. Combine the whitened covariances of all sub-windows into a block diagonal matrix
    final_covariances = np.array([
        block_diag(*[enhanced_covariances[j][i] for j in range(N)]) 
        for i in range(num_samples)
    ]) # Shape: (n_trials, n_channels * K * N, n_channels * K * N)

    return final_covariances, whiten_filter

import numpy as np
from scipy.linalg import block_diag
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

def enhanced_cov(X, t_win: list[tuple[int, int]], tau: int or list[int],  # type: ignore
                 whiten_filter=None, estimator='cov', metric='euclid', sample_weight=None):
    """
    Compute enhanced covariance matrices with whitening based on given time windows and FIR filter time delays.

    Parameters:
    X (ndarray): Input EEG data of shape (n_trials, n_channels, n_timepoints).
    t_win (list of tuples): List of (start, end) time window tuples, where each tuple specifies a window.
    tau (int or list of ints): Time delays (must be positive integers).
    estimator (str): Covariance estimator to use.
    metric (str): Distance metric to use.
    whiten_filter (ndarray): Whitening filter for each sub-window.
    sample_weight (ndarray): Sample weights.

    Returns:
    final_covariances: Enhanced whitened covariance matrices of shape (n_trials, n_channels * K * N, n_channels * K * N).
    whiten_filter: Whitening filter for each sub-window, shape (N, n_channels * K, n_channels * K).
    """
    num_samples, num_channels, num_time_points = X.shape
    is_train = whiten_filter is None  # Determine if in training mode
    if is_train:
        whiten_filter = []

    # Ensure tau is a list if it's a scalar
    if tau is None:  # If tau is not specified, use [0]
        tau = [0]
    if isinstance(tau, int):
        tau = [tau]
    K = len(tau)
    
    if t_win is None:  # If t_win is not specified, use the whole signal as a single time window
        t_win = [(0, num_time_points)]
    if isinstance(t_win, (list, tuple)) and len(t_win) == 2 and all(isinstance(x, int) for x in t_win):
        t_win = [tuple(t_win)]  # Convert to list of tuples
    if not (isinstance(t_win, (list, tuple)) and all(isinstance(win, (list, tuple)) and len(win) == 2 for win in t_win)):
        raise ValueError("t_win must be a list or tuple of (start, end) pairs.")
    N = len(t_win)  # Number of specified time windows

    # 1. Extract and process each time window specified in t_win
    sub_signals = []
    for start, end in t_win:
        if end > num_time_points:
            raise ValueError("End of time window exceeds available time points.")
        sub_signal = X[:, :, start:end]  # Extract specified sub-window (n_trials, n_channels, t_win)

        # 2. Apply time delays to each sub-window and concatenate along the channel dimension
        delayed_signals = []
        for delay in tau:
            if delay == 0:
                delayed_signal = sub_signal.copy()  # No shift needed
            else:
                delayed_signal = np.zeros_like(sub_signal)
                delayed_signal[:, :, delay:] = sub_signal[:, :, :-delay]  # Apply delay
                delayed_signal[:, :, :delay] = 0  # Zero padding for the first 'delay' points
            delayed_signals.append(delayed_signal)

        # Concatenate delayed signals along the channel dimension for the current time window
        sub_signals.append(np.concatenate(delayed_signals, axis=1))  # Shape: (n_trials, n_channels * K, t_win)

    # 3. Compute and whiten covariance matrices for each sub-window
    enhanced_covariances = []
    for i, sub_signal in enumerate(sub_signals):
        # Calculate covariance matrices for sub-signal
        cov_matrices = covariances(sub_signal, estimator=estimator)  # Shape: (n_trials, n_channels * K, n_channels * K)
        
        # Trace normalization
        cov_matrices /= np.maximum(
            np.trace(cov_matrices, axis1=1, axis2=2)[:, np.newaxis, np.newaxis], 
            1e-10
        )

        # Whitening step
        if is_train:
            mean_cov = mean_covariance(cov_matrices, metric=metric, sample_weight=sample_weight)
            whiten_filter.append(invsqrtm(mean_cov))  # Calculate whitening filter for each sub-window
        whitened_cov_matrices = np.einsum('ij,bjk,kl->bil', whiten_filter[i].T, cov_matrices, whiten_filter[i])
        
        enhanced_covariances.append(whitened_cov_matrices)
        
    # 4. Combine the whitened covariances of all sub-windows into a block diagonal matrix
    final_covariances = np.array([
        block_diag(*[enhanced_covariances[j][i] for j in range(N)]) 
        for i in range(num_samples)
    ])  # Shape: (n_trials, n_channels * K * N, n_channels * K * N)

    return final_covariances, whiten_filter


class CTSSP(BaseEstimator, TransformerMixin):
    """Common Temporal-Spatial-Spectral Patterns.

    Parameters
    ----------
    nfilter : int
        Number of filters to extract.
    t_win : list of tuples, optional, default: None
        List of (start, end) time window tuples, where each tuple specifies a window.
        if None, use the whole signal as a single time window.
    tau : int or list, optional, default: None
        Time delays (must be positive integers). e.g. [0, 1]
    cov_method : str, optional, default: 'cov'
        Covariance estimator to use.
    metric : str, optional, default: 'euclid'
        Distance metric to use.
    log : bool, optional, default: True
        If True, return log-variance instead of covariance.
    
    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_filters)
        Spatial filters.
    patterns_ : ndarray, shape (n_channels, n_filters)
        Spatial patterns.
    Mrct_ : ndarray, shape (n_channels, n_channels)
        Recentering matrix.
    """

    def __init__(self, 
                 nfilter=10,
                 t_win=None, 
                 tau=None,
                 cov_method='cov',
                 metric='euclid',
                 log=True,
                 ):
        self.nfilter = nfilter
        self.t_win = t_win
        self.tau = tau
        self.cov_method = cov_method
        self.metric = metric    
        self.log = log
        self.classes_ = None
        self.filters_ = None
        self.patterns_ = None
        self.Mrct_ = None
        
    def _get_recenter(self, X, sample_weight=None):
        M = mean_covariance(X, self.metric, sample_weight)
        return invsqrtm(M)
    
    def _recenter(self, X, filters):
        return np.einsum('ij,bjk,kl->bil', filters.T, X, filters)
       
    def fit(self, X, y, sample_weight=None):
        """Train CTSSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_timepoints)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : object
            Returns the instance itself.
            
        """
        if not isinstance(self.nfilter, int):
            raise TypeError('nfilter must be an integer')
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_timepoints')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')
        
        sample_weight = check_weights(sample_weight, len(y))

        classes = np.unique(y)
        self.classes_ = classes
        
        Cov, self.Mrct_ = enhanced_cov(
            X, 
            self.t_win, 
            self.tau, 
            estimator=self.cov_method, 
            metric=self.metric, 
            sample_weight=sample_weight
            )
        
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(Cov[y == c], self.metric, sample_weight=sample_weight[y == c]))
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
    
    def transform(self, X):
        """Apply CTSSP spatial filters to new data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_timepoints)
            Set of covariance matrices.

        Returns
        -------
        Out : ndarray, shape (n_trials, n_filters) or \
                ndarray, shape (n_trials, n_filters, n_filters)
            Set of spatialy filtered log-variance or covariance, depending on
            the 'log' input parameter.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_timepoints')
        if self.filters_ is None:
            raise ValueError('The model is not fitted yet.')
        
        Cov, _ = enhanced_cov(
            X, 
            self.t_win, 
            self.tau, 
            self.Mrct_, 
            estimator=self.cov_method, 
            metric=self.metric, 
            )
        Cov_filt = self.filters_ @ Cov @ self.filters_.T

        # if logvariance
        if self.log:
            out = np.zeros(Cov_filt.shape[:2])
            for i, x in enumerate(Cov_filt):
                out[i] = np.log(np.diag(x))
            return out
        else:
            return Cov_filt

class TRCTSSP(CTSSP): 
    """Weighted Tikhonov-regularized Common Temporal-Spatial-Spectral Patterns.

    Parameters
    ----------
    nfilter : int
        Number of filters to extract.
    t_win : list of tuples, optional, default: None
        List of (start, end) time window tuples, where each tuple specifies a window.
        if None, use the whole signal as a single time window.
    tau : int or list, optional, default: None
        Time delays (must be positive integers). e.g. [0, 1]
    rho : float, optional, default: 1
        Regularization parameter.
    cov_method : str, optional, default: 'cov'
        Covariance estimator to use.
    metric : str, optional, default: 'euclid'
        Distance metric to use.
    log : bool, optional, default: True
        If True, return log-variance instead of covariance.
    
    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_filters)
        Spatial filters.
    patterns_ : ndarray, shape (n_channels, n_filters)
        Spatial patterns.
    Mrct_ : ndarray, shape (n_channels, n_channels)
        Recentering matrix.
    """
     
    def __init__(self, 
                 nfilter=10,
                 t_win=None, 
                 tau=None,
                 rho=1,
                 cov_method='cov',
                 metric='euclid',
                 log=True,
                 ):
        self.nfilter = nfilter
        self.t_win = t_win
        self.tau = tau
        self.rho = rho
        self.cov_method = cov_method
        self.metric = metric    
        self.log = log
        self.classes_ = None
        self.filters_ = None
        self.patterns_ = None
        self.Mrct_ = None   
    
    def fit(self, X, y, sample_weight=None):
        """Train spatial filters.   
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_timepoints)
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
        
        classes = np.unique(y)
        self.classes_ = classes
        assert len(classes) == 2, "Can only do 2-class TRCTSSP"
        
        Cov, self.Mrct_ = enhanced_cov(
            X, 
            self.t_win, 
            self.tau, 
            estimator=self.cov_method, 
            metric=self.metric, 
            sample_weight=sample_weight
            )

        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(Cov[y == c], self.metric, sample_weight=sample_weight[y == c]))
        C = np.array(C)

        # regularize CSP
        evals = [[], []]
        evecs = [[], []]
        Creg = C[1] + np.eye(C[1].shape[0]) * self.rho
        evals[1], evecs[1] = eigh(C[0], Creg)
        Creg = C[0] + np.eye(C[0].shape[0]) * self.rho
        evals[0], evecs[0] = eigh(C[1], Creg)
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

# only for 2 classes
class SBL_CTSSP(BaseEstimator, ClassifierMixin):
    """Sparse Bayesian Learning for Common Temporal-Spatial-Spectral Patterns.

    Parameters
    ----------
    t_win : list of tuples, optional, default: None
        List of (start, end) time window tuples, where each tuple specifies a window.
        if None, use the whole signal as a single time window.
    tau : int or list, optional, default: None
        Time delays (must be positive integers). e.g. [0, 1]
    cov_method : str, optional, default: 'cov'
        Covariance estimator to use.
    metric : str, optional, default: 'euclid'
        Distance metric to use.
    epoch : int, optional, default: 5000
        Number of epochs to train the model.
    device : str, optional, default: 'cpu'
        Device to use for training.
    
    Attributes
    ----------
    Mrct_ : ndarray, shape (n_channels, n_channels)
            Recentering matrix.
    W     : Estimated low-rank weight matrix. [K*N*n_channels, K*C*n_channels].
            where K is the number of time delays, N is the number of time windows.
    alpha : Classifier weights. [L, 1].
    V     : Spatio-temporal filter matrix. [K*N*n_channels, L].
            Each column of V represents a spatio-temporal filter.
    """

    def __init__(self, 
                 t_win=None, 
                 tau=None,
                 cov_method='cov',
                 metric='euclid',
                 epoch=5000,
                 device='cpu',
                 ):

        self.t_win = t_win
        self.tau = tau
        self.cov_method = cov_method
        self.metric = metric
        self.epoch = epoch
        self.device = device
        self.classes_ = None
        self.filters_ = None
        self.patterns_ = None
        self.Mrct_ = None
        self.W = None
        self.alpha = None
        self.V = None
        
    def _get_vector(self, X):    
        """
        Compute the vectorized matrix of the augmented empirical covariance matrix.

        Parameters:
        X (ndarray): Enhanced covariance matrix of shape (n_trials, n_channels * N * K, n_channels * N * K).
                     where N is the number of time windows, and K is the number of time delays.
        sample_weight (ndarray): Sample weights.

        Returns:
        ndarray: Column-wise vector. shape (n_trials, (n_channels * N * K)^2)
        where N is the number of time windows, and K is the number of time delays.
        """

        X = logm(X) # logarithm transformation
        R = np.reshape(X, (X.shape[0], -1)) # column-wise vectorization
        return R
        
    def fit(self, X, y, sample_weight=None):
        """Train CTSSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_timepoints)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : object
            Returns the instance itself.
            
        """

        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_timepoints')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')
        
        classes = np.unique(y)
        self.classes_ = classes
        assert len(classes) == 2, "Can only do 2-class SBL_CTSSP"
        
        Y = np.where(y == np.unique(y)[0], 1, -1)# Convert labels to -1 and 1
        Y = np.array(Y).reshape(-1, 1)
        
        sample_weight = check_weights(sample_weight, len(y))
        
        Cov, self.Mrct_ = enhanced_cov(
            X, 
            self.t_win, 
            self.tau, 
            estimator=self.cov_method, 
            metric=self.metric, 
            sample_weight=sample_weight
            )
        R = self._get_vector(Cov)
        self.W, self.alpha, self.V = sblest_kernel(
            R, Y, epoch=self.epoch, device=self.device)
        
        return self
    
    def predict(self, X):
        if self.W is None:
            raise ValueError("Model is not trained yet. Please call 'fit' with \
                             appropriate arguments before calling 'predict'.")
        Cov, _ = enhanced_cov(
            X, 
            self.t_win, 
            self.tau, 
            self.Mrct_, 
            estimator=self.cov_method, 
            metric=self.metric, 
            )
        R = self._get_vector(Cov)
        vec_W = self.W.T.flatten()
        predict_Y = R @ vec_W
        return np.where(predict_Y > 0, self.classes_[0], self.classes_[1])
    



