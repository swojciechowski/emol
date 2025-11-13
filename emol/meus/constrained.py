import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class ConstrainedBinarySampling(Sampling):
    def __init__(self, n_max, *args, **kwargs):
        self.n_max = n_max
        super().__init__(*args, **kwargs)

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[: self.n_max]
            X[k, I] = True

        return X


class ConstrainedBinaryCrossover(Crossover):
    def __init__(self, n_max, *args, **kwargs):
        self.n_max = n_max
        super().__init__(n_parents=2, n_offsprings=1, *args, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = self.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class ConstrainedBinaryMutation(Mutation):
    def __init__(self, n_max, *args, **kwargs):
        self.n_max = n_max
        super().__init__(*args, **kwargs)

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X

def get_constrained_operators(n_max):
    return {
        "sampling": ConstrainedBinarySampling(n_max),
        "crossover": ConstrainedBinaryCrossover(n_max),
        "mutation": ConstrainedBinaryMutation(n_max),
    }