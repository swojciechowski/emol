import numpy as np

class DummyFitting:
    def __init__(self, test_size=1.0, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def initialize(self, X, y):
        self.X = X
        self.y = y

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
