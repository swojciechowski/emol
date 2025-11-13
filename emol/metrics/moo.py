from scipy.spatial.distance import cdist
import numpy as np
from pymoo.indicators import hv


def non_dominated_solutions(pareto, distinct=False):
    is_efficient = np.zeros(len(pareto), dtype=bool)

    for i in range(len(pareto)):
        this_cost = pareto[i, :]

        at_least_as_good = np.logical_not(np.any(pareto < this_cost, axis=1))
        any_better = np.any(pareto > this_cost, axis=1)

        dominated_by = np.logical_and(at_least_as_good, any_better)

        if distinct and np.any(is_efficient):
            if np.any(np.all(pareto[is_efficient] == this_cost, axis=1)):
                continue

        if not (np.any(dominated_by[:i]) or np.any(dominated_by[i + 1 :])):
            is_efficient[i] = True

    return is_efficient


def GD(pareto, ideal, p=2):
    dist_mat = cdist(pareto, ideal)
    return np.mean(np.min(dist_mat, axis=1))


def HV(front, ref_point, minimize=False):
    ind = hv.HV(ref_point=ref_point)
    return ind(-1 * np.array(front))
