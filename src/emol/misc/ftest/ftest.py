import numpy as np
from scipy import stats

def F_test(a, b, cv=(5, 2)):
    n_repeats, k_folds = cv

    delta = a.reshape(n_repeats, k_folds) - b.reshape(n_repeats, k_folds)
    fold_mean = np.mean(delta, axis=-1)
    var = np.sum(np.power(delta - fold_mean[:, None], 2), axis=-1)

    f_stat = np.sum(np.power(delta, 2)) / (2 * np.sum(var))
    pvalue = stats.f.sf(f_stat, k_folds, k_folds * n_repeats)

    return f_stat, pvalue