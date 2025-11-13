from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifierEx(KNeighborsClassifier):
    def __init__(self, n_neighbors = 5, *, weights = "uniform", algorithm = "auto", leaf_size = 30, p = 2, metric_params = None, n_jobs = None):
        super().__init__(n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric="minkowski", metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight):
        # self.set_params({"metric_params": sample_weight})
        self.metric_params = dict(w=sample_weight)
        return super().fit(X, y)