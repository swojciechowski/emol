import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin


class SDKNN(BaseEstimator, ClassifierMixin):
    """
    Scaled Distance K-Nearest Neighbors
    """
    def __init__(self, k_neighbors=5, p=2, use_distance=False):
        self.k_neighbors = k_neighbors
        self.p = p
        self.use_distance = use_distance

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

        if self.sample_weight_ is not None:
            distances = distances / self.sample_weight_

        knn = np.argsort(distances, axis=1)[:, :self.k_neighbors]
        kn_distances = np.take_along_axis(distances, knn, axis=1)

        if self.use_distance:
            weights = 1 / kn_distances

            pred = np.array([
                np.bincount(self.y_[n], weights=w, minlength=len(self.classes_))
                for w, n in zip(weights, knn)
            ])

            pred /= np.sum(pred, axis=1)[:, None]

        else:
            pred = np.array([
                np.bincount(self.y_[n], minlength=len(self.classes_)) for n in knn
            ])

            pred = pred / self.k_neighbors

        return pred
    