import numpy as np

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

class BootstrapFitting:
    def __init__(self, bootstrap_size=1.0, shrinkage=1.0, random_state=None):
        self.bootstrap_size = bootstrap_size
        self.shrinkage = shrinkage
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def make_boostrap(self):
        strategy = {
            cl: c_size + int(c_size * self.bootstrap_size)
            for cl, c_size in Counter(self.y).items()
        }

        X_, y_ = RandomOverSampler(
            sampling_strategy=strategy, shrinkage=self.shrinkage, random_state=self._rs
        ).fit_resample(self.X, self.y)

        # Drop original samples
        return X_[len(self.X):], y_[len(self.X):]

    def initialize(self, X, y):
        self.X = X
        self.y = y
        self.bootstrap = self.make_boostrap()

        return self.X, self.y

    def update(self):
        self.bootstrap = self.make_boostrap()

    def __iter__(self):
        yield np.arange(len(self.X)), *self.bootstrap

