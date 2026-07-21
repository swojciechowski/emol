import numpy as np

def saw(indicators, weights=None):
    n_criteria = indicators.shape[1]

    # Adjust weight if not given
    weights = weights if weights is not None else np.ones(n_criteria) / n_criteria

    # Normalize
    si = indicators / np.min(indicators, axis=0)

    # Sum
    sm = np.sum(si * weights, axis=1)

    return np.argmax(sm)
