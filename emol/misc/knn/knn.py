import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neighbors import KDTree

import time

class KNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors
    """

    def __init__(self, k_neighbors=5, p=2):
        self.k_neighbors = k_neighbors
        self.p = p

    def fit(self, X, y, sample_weight=None):
        self.X_, self.y_, self.sample_weight_ = (
            np.copy(X),
            np.copy(y),
            np.copy(sample_weight) if sample_weight is not None else None,
        )

        self.tree = KDTree(X)

        self.classes_ = np.unique(y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # distances = cdist(X, self.X_, p=self.p, metric="minkowski")
        # knn = np.argsort(distances, axis=1)[:, : self.k_neighbors]
        distances, knn = self.tree.query(X)

        if self.sample_weight_ is None:
            pred = np.array(
                [np.bincount(self.y_[n], minlength=len(self.classes_)) for n in knn]
            )

            pred = pred / self.k_neighbors

        else:
            pred = np.array(
                [
                    np.bincount(
                        self.y_[n],
                        weights=self.sample_weight_[n],
                        minlength=len(self.classes_),
                    )
                    for n in knn
                ]
            )

            pred /= np.sum(pred, axis=1)[:, None]

        return pred
