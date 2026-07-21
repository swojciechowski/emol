from itertools import combinations

import numpy as np

DEFAULT_LIM = 100
DEFAULT_N = 200

def fbeta(tpr, ppv, beta=1.0):
    if not hasattr(beta, "__iter__"):
        beta = np.array([beta])

    beta_sqr = np.power(beta, 2)

    return np.nan_to_num(
        (1 + beta_sqr)
        * ppv[..., np.newaxis]
        * tpr[..., np.newaxis]
        / ((beta_sqr * ppv[..., np.newaxis]) + tpr[..., np.newaxis])
    )


def fbeta_zero_point(TPR_A, PPV_A, TPR_B, PPV_B):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(
            (TPR_A * TPR_B * (PPV_B - PPV_A)) / (PPV_A * PPV_B * (TPR_A - TPR_B))
        )


def fbeta_crossing_mat(series):
    c_mat = np.full((len(series), len(series)), np.nan, dtype=float)

    # Check each pair
    for s1, s2 in combinations(range(len(series)), 2):
        c_pnt = fbeta_zero_point(*series[s1], *series[s2])

        c_mat[s1, s2] = c_pnt
        c_mat[s2, s1] = c_pnt

    return c_mat


def combined_curve(series):
    return np.max(series, axis=0)


def crossing_approximations(series_a, series_b, space=None):
    cmp = series_a > series_b
    change_indices = np.where(cmp[:-1] != cmp[1:])[0] + 1

    return space[change_indices] if space is not None else change_indices


def fbeta_best(series, names, lim=DEFAULT_LIM, N=DEFAULT_N):
    betas = np.geomspace(1 / lim, lim, N)

    f_betas = fbeta(series[..., 0], series[..., 1], beta=betas)

    if len(f_betas.shape) > 1:
        f_betas = f_betas.mean(axis=0)

    best = np.argmax(f_betas, axis=0)

    _, c_i = np.unique(best, return_index=True)

    candidates = best[np.sort(c_i)]
    return np.array(names)[candidates]

def allmax(A):
    return np.argwhere(A == np.nanmax(A)).flatten()

def allmin(A):
    return np.argwhere(A == np.nanmin(A)).flatten()

def fbeta_rank(series, precision=0.0):
    ret_mat = []

    cross_mat = fbeta_crossing_mat(series)

    split_point = - np.inf
    candidates = np.argwhere(series[:, 1] == np.max(series[:, 1])).flatten()
    candidate = candidates[np.argsort(series[candidates, 0])[-1]]

    while True:
        print(split_point, candidate)
        ret_mat.append([split_point, candidate])

        splits = cross_mat[candidate]
        splits[splits <= (split_point)] = np.nan

        if np.all(np.isnan(splits)):
            break

        split_point = np.nanmin(splits)

        if split_point == np.inf:
            break

        candidates = np.argwhere(splits == split_point).flatten()
        candidate = candidates[np.argsort(series[candidates, 0])[-1]]

    return ret_mat
