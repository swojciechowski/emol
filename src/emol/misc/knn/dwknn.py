import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin


class DWKNN(BaseEstimator, ClassifierMixin):
    """
    Distance Weighted K-Nearest Neighbors
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

        self.classes_ = np.unique(y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        distances = cdist(X, self.X_, p=self.p, metric='minkowski')
        knn = np.argsort(distances, axis=1)[:, :self.k_neighbors]
        kn_distances = np.take_along_axis(distances, knn, axis=1)

        if self.sample_weight_ is None:
            weights = 1 / kn_distances
        else:
            weights = self.sample_weight_[knn] / kn_distances

        pred = np.array([
            np.bincount(self.y_[n], weights=w, minlength=len(self.classes_))
            for w, n in zip(weights, knn)
        ])

        pred /= np.sum(pred, axis=1)[:, None]

        return pred