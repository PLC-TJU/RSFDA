from metabci.brainda.algorithms.manifold import get_recenter, get_rescale, get_rotate, recenter, rescale, rotate
from sklearn.base import BaseEstimator, TransformerMixin

class RCT(BaseEstimator, TransformerMixin):
    def __init__(self, mean_method='euclid', cov_method='cov'):
        self.mean_method = mean_method
        self.cov_method = cov_method

    def fit(self, X, y=None):

        self.center_ = get_recenter(
            X, 
            cov_method=self.cov_method, 
            mean_method=self.mean_method
            )

        return self

    def transform(self, X):
        X = recenter(X, self.center_)
        return X    