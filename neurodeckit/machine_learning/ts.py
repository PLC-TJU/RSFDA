
import numpy as np
from sklearn.base import is_classifier

from pyriemann.tangentspace import TangentSpace, FGDA
from pyriemann.classification import FgMDM, MDM, TSclassifier

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.utils import check_metric

from .base import chesk_sample_weight, recursive_reference_center
from ..utils import combine_processes

class TS_online(TangentSpace):
    def __init__(self, metric='riemann', tsupdate=None, min_tracked=6):
        """ Tangent space projection with online update.

        Parameters
        ----------
        metric : str, default='riemann'
            Metric to compute the mean covariance matrix.
        tsupdate : str, default=None
            Online update method for tangent space projection.
            If 'online', it uses the online method.
            If 'offline', it uses the offline method.
            otherwise, it wont use any update method.
        
        """
        self.metric = metric
        self.tsupdate = tsupdate
        self.min_tracked = min_tracked
        self._n_tracked = 0
        
    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean, self.metric_map = check_metric(
            self.metric, ["mean", "map"]
        )
        self._check_reference_points(X)
        
        if self.tsupdate == 'offline':
            Cr = mean_covariance(X, metric=self.metric_mean) 

            TsX = []
            for i in range(X.shape[0]):
                if i == 0:
                    temp_reference_ = X[i]
                else:
                    temp_reference_ = recursive_reference_center(
                        temp_reference_, X[i], alpha=1/(i+1), metric=self.metric_mean)
                
                if i >= self.min_tracked-1:
                    self.reference_ = temp_reference_
                
                Cr = self.reference_
                TsX.append(tangent_space(X[i], Cr, metric=self.metric_map))

            return np.array(TsX)

        elif self.tsupdate == 'online':
            # Cr = mean_covariance(X, metric=self.metric_mean)
            if self._n_tracked == 0:
                self.temp_reference_ = X
                
            self._n_tracked += 1
            
            if self._n_tracked > 1:
                alpha = 1 / self._n_tracked # alpha为新样本权重，1-alpha为旧样本权重
                # XX = np.concatenate([self.temp_reference_[np.newaxis,:],X],axis=0) 
                # self.temp_reference_ = mean_covariance(XX, metric=self.metric_mean, sample_weight=np.array([1-alpha,alpha]))
                self.temp_reference_ = recursive_reference_center(self.temp_reference_, X, alpha=alpha, metric=self.metric_mean)

            if self._n_tracked >= self.min_tracked:
                self.reference_ = self.temp_reference_
            
            Cr = self.reference_
        else:
            Cr = self.reference_
            
        return tangent_space(X, Cr, metric=self.metric_map)
    
class FGDA_online(FGDA):
    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point and the FLDA.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : FGDA instance
            The FGDA instance.
        """
        self._ts = TS_online(metric=self.metric, tsupdate=self.tsupdate)
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        covs : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices after filtering.
        """
        self._ts = TS_online(metric=self.metric, tsupdate=self.tsupdate)
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)

class FgMDM_online(FgMDM):
    def fit(self, X, y, sample_weight=None):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self.classes_ = np.unique(y)

        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA_online(metric=self.metric, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y, sample_weight=sample_weight)
        self._mdm.fit(cov, y, sample_weight=sample_weight)
        self.classes_ = self._mdm.classes_
        return self

class TSclassifier_online(TSclassifier):
    def __init__(self, metric="riemann", tsupdate=False,
                 clf=None, memory=None):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf
        self.memory = memory
        
    def fit(self, X, y, sample_weight=None):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TSclassifier instance
            The TSclassifier instance.
        """
        if not is_classifier(self.clf):
            raise TypeError('clf must be a classifier')
        
        self.classes_ = np.unique(y)

        ts = TS_online(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = combine_processes(ts, self.clf, memory=self.memory)
        sample_weight_dict = {}
        for step in self._pipe.steps:
            step_name = step[0]
            step_pipe = step[1]
            if chesk_sample_weight(step_pipe):
                    sample_weight_dict[step_name + '__sample_weight'] = sample_weight
        self._pipe.fit(X, y, **sample_weight_dict)
        return self