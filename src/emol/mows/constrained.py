import numpy as np

from pymoo.core.repair import Repair

class NormalizingRepair(Repair):
    def _do(self, problem, X, **kwargs):
        return X / X.sum(axis=1, keepdims=True)

def get_constrained_operators():
    return {
        "repair": NormalizingRepair(),
    }