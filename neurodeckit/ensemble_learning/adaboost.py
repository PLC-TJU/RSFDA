from sklearn.ensemble import AdaBoostClassifier

# import warnings
# from abc import ABCMeta, abstractmethod
# from numbers import Integral, Real

import numpy as np
from scipy.special import xlogy

# from sklearn.base import (
#     ClassifierMixin,
#     RegressorMixin,
#     _fit_context,
#     is_classifier,
#     is_regressor,
# )
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.utils import _safe_indexing, check_random_state
# from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
# from sklearn.utils.extmath import softmax, stable_cumsum
# from sklearn.utils.metadata_routing import (
#     _raise_for_unsupported_routing,
#     _RoutingNotSupportedMixin,
# )
# from sklearn.utils.validation import (
#     _check_sample_weight,
#     _num_samples,
#     check_is_fitted,
#     has_fit_parameter,
# )
# from sklearn.ensemble._base import BaseEnsemble

from ..transfer_learning import decode_domains

class AdaBoost(AdaBoostClassifier):
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        _, y_true, _ = decode_domains(X, y)
        incorrect = y_predict != y_true

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        _, y_true, _ = decode_domains(X, y)
        incorrect = y_predict != y_true

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error