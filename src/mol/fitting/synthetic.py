import numpy as np

from collections import Counter
from imblearn.over_sampling import RandomOverSampler


class SyntheticFitting:
    def __init__(
        self, test_size=1.0, shrinkage=1.0, dynamic=False, random_state=None
    ):
        self.test_size = test_size
        self.shrinkage = shrinkage
        self.dynamic = dynamic
        self.random_state = random_state

    def make_test(self):
        strategy = {
            cl: c_size + int(c_size * self.test_size)
            for cl, c_size in Counter(self.y).items()
        }

        X_, y_ = RandomOverSampler(
            sampling_strategy=strategy, shrinkage=self.shrinkage, random_state=self._rs
        ).fit_resample(self.X, self.y)

        # Drop original samples
        return X_[len(self.X) :], y_[len(self.X) :]

    def initialize(self, X, y):
        self.X = X
        self.y = y

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

        self.test = self.make_test()

        return self.X, self.y

    def update(self):
        self.test = self.make_test()

    def __iter__(self):
        yield np.arange(len(self.X)), *self.test

        if self.dynamic:
            self.update()
