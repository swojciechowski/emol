import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, prior_proba=False, random_state=None):
        self.prior_proba = prior_proba
        self.random_state = random_state

    def fit(self, X, y):
        self._classes, c_counts = np.unique(y, return_counts=True)

        if self.prior_proba:
            self._probas = c_counts / np.sum(c_counts) 
        else:
            self._probas = np.ones(len(self._classes)) / len(self._classes)

        self._random_word = np.random.RandomState(self.random_state).randint(0xffffffff)

        return self

    def predict(self, X):
        predictions = []

        for row in X:
            row_hash = (hash(tuple(row)) & 0xffffffff) ^ self._random_word
            predictions.append(
                np.random.RandomState(row_hash).choice(self._classes)
            )

        return np.array(predictions)
