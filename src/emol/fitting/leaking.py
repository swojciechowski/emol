import numpy as np


class LeakingFitting:
    def __init__(self, test_size=1.0, dynamic=False, random_state=None):
        self.test_size = test_size
        self.dynamic = dynamic
        self.random_state = random_state

    def __copy__(self):
        return type(self)(self.test_size, self.dynamic, self.random_state)

    def initialize(self, X, y):
        self.X = X
        self.y = y

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

        self.test = self.make_test()

        return X, y

    def make_test(self):
        if self.test_size >= 1.0:
            return np.arange(len(self.X))

        return self._rs.choice(
            len(self.X), size=int(len(self.X) * self.test_size), replace=False
        )

    def update(self):
        self.test = self.make_test()

    def __iter__(self):
        yield np.arange(len(self.X)), self.X[self.test], self.y[self.test]

        if self.dynamic:
            self.update()
