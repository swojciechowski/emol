import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BayesRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, prior_proba=False, random_state=None):
        self.prior_proba = prior_proba
        self.random_state = random_state

    def fit(self, X, y):
        self._classes, c_counts = np.unique(y, return_counts=True)

        if self.prior_proba:
            self._probas = c_counts / np.sum(c_counts) 
        else:
            self._probas = np.ones(len(self._classes)) / len(self._classes)

        return self

    def predict(self, X):
        rng = np.random.RandomState(self.random_state)
        return rng.choice(self._classes, size=len(X), p=self._probas)
