import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

class CrossValidationFitting:
    def __init__(self, n_repeats=1, n_splits=2, stratified=True, random_state=None):
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def initialize(self, X, y):
        self.X = X
        self.y = y
        self.cv = self.make_cv()

        return X, y

    def make_cv(self):
        cv_class = RepeatedStratifiedKFold if self.stratified else RepeatedKFold
        seed = self._rs.randint(0xffffffff) # Generate new seed, as it will assure same split if not dynamic
        return cv_class(n_repeats=self.n_repeats, n_splits=self.n_splits, random_state=seed)

    def update(self):
        self.cv = self.make_cv()

    def __iter__(self):
        for train, test in self.cv.split(self.X, self.y):
            yield train, self.X[test], self.y[test]
