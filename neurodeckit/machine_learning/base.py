
from typing import List, Tuple, Optional
from sklearn.pipeline import Pipeline
from inspect import signature
from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.base import invsqrtm
from scipy.signal import cheby1, cheb1ord

def generate_filterbank(
    passbands: List[Tuple[float, float]],
    stopbands: List[Tuple[float, float]],
    srate: int,
    order: Optional[int] = None,
    rp: float = 0.5,
):
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)

        filterbank.append(sos)
    return filterbank

def chesk_sample_weight(clf):
    if isinstance(clf, Pipeline):
        return chesk_sample_weight(clf.steps[-1][1])
    else:
        fit_method = getattr(clf, 'fit')
        params = signature(fit_method).parameters
        return 'sample_weight' in params

def recursive_reference_center(reference_old, X_new, alpha, metric='riemann'):
    """Recursive reference centering.

    Parameters
    ----------
    reference_old : ndarray, (1, n_channels, n_channels) or (n_channels, n_channels)
        The reference matrix to be updated.
    X_new : ndarray, shape (1, n_channels, n_channels) or (n_channels, n_channels)
        The new matrices to be centered.
    alpha : float
        The weight to assign to the new samples.
    metric : str, default="riemann"
        The metric to use for the geodesic distance.

    Returns
    -------
    reference_new : ndarray, shape (n_channels, n_channels)
        The updated reference matrix. 
    """
    X_new = X_new.copy()
    reference_old = reference_old.copy()
    X_new = X_new.reshape((-1, *X_new.shape[-2:]))
    X_new = X_new.mean(axis=0, keepdims=False)
    if reference_old.shape[0] == 1:
        reference_old = reference_old[0]
    C = geodesic(reference_old, X_new, alpha, metric=metric)
    reference_new = invsqrtm(C)
    reference_new = reference_new.reshape(reference_old.shape)
    return reference_new
    
  