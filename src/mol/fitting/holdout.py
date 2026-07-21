import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


class HoldOutFitting:
    def __init__(self, test_size=0.2, stratified=True, random_state=None):
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

    def __copy__(self):
        return type(self)(self.test_size, self.stratified, self.random_state)

    def initialize(self, X, y):
        self.X = X
        self.y = y

        split = (
            StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            if self.stratified
            else ShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
        )

        self.train, self.test = next(split.split(X, y))
        return X[self.train], y[self.train]

    def __iter__(self):
        yield np.arange(len(self.train)), self.X[self.test], self.y[self.test]
