import numpy as np

from pymoo.core.mutation import Mutation


class UniformMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        changed = np.random.random(X.shape) < prob_var
        mutations = np.random.uniform(problem.xl[0], problem.xu[0], np.sum(changed))
        Xp[changed] = mutations
        return Xp
