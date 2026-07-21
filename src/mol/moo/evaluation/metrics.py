import numpy as np
from scipy.spatial.distance import cdist

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

# Generational Distance
def GD(front, ideal, ideal_ref=False, p=2):
    if len(ideal.shape) == 1:
        ideal = np.array([ideal])

    # If ideal is reference point, select only those dominated by it.
    if ideal_ref:
        front = front[np.all(front < ideal, axis=1)]

    # Return 0 if front is empty
    if front.size == 0:
        return 0

    dist_mat = cdist(front, ideal, metric='minkowski', p=p)
    return np.mean(np.min(dist_mat, axis=1))


# Hyper-volume
def HV(front, ref_point):
    dominating = np.all(front > ref_point, axis=1)
    d_front = front[dominating]
    p_dominating = non_dominated_solutions(d_front)
    d_front = d_front[p_dominating]
    sorted_front = d_front[np.argsort(d_front[:, 1])]

    areas = []
    for a, b in zip(np.vstack([ref_point, sorted_front]), sorted_front):
        areas.append((ref_point[0] - b[0]) * (a[1] - b[1]))

    return np.sum(areas)


# Overall Non-dominated vector generation ratio
def ONVGR(front):
    return np.sum(non_dominated_solutions(front)) / len(front)


# Spacing
def SP(front):
    D = cdist(front, front, metric="cityblock")
    D[np.diag_indices(len(D))] = np.inf
    d = np.min(D, axis=0)
    dm = np.mean(d)

    # Check for possible 0
    s_sum = np.sum((dm - d) ** 2) / (len(front) - 1)
    return np.sqrt(s_sum) if s_sum > 0 else 0


def efficiency_matrix(front, ref_point, weak=False):
    cmp = (front >= ref_point).T if weak else (front > ref_point).T
    cnt = np.bincount(2 * cmp[0] + cmp[1], minlength=4)
    return cnt.reshape(2, 2)


def DR(front, ref_point, strict=False):
    em = efficiency_matrix(front, ref_point, weak=strict)
    return em[1, 1] / np.sum(em) if strict else (np.sum(em) - em[0, 0]) / np.sum(em)
