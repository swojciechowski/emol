import numpy as np

from pymoo.core.problem import Problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.max_gen import MaximumGenerationTermination

from pymoo.operators.sampling.rnd import BinaryRandomSampling, FloatRandomSampling
from pymoo.operators.crossover.spx import SinglePointCrossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation


class ModelEvaluationProblem(Problem):
    def __init__(self, eval_cb, n_var, n_obj=-1, xl=0, xu=1, vtype=bool):
        self.eval_cb = eval_cb
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=vtype)

    def _calc_pareto_front(self, *args, **kwargs):
        return np.array([[-1.0] * self.n_obj])

    def _evaluate(self, S, out, *args, **kwargs):
        scores, models = self.eval_cb(S)
        out["F"] = scores
        out["A"] = models


DEFAULT_SEED = 90210

DEFAULT_MOO_SOLVER = NSGA2
DEFAULT_SOO_SOLVER = GA

DEFAULT_SOLVER_PARAMS = {
    "population_size": 100,
}

DEFAULT_FLOAT_OPERATORS = {
    "sampling": FloatRandomSampling(),
    "crossing": SimulatedBinaryCrossover(),
    "mutation": PolynomialMutation(),
}

DEFAULT_BOOL_OPERATORS = {
    "sampling": BinaryRandomSampling(),
    "crossing": SinglePointCrossover(),
    "mutation": BitflipMutation(),
}

DEFAULT_TERMINATION = MaximumGenerationTermination
DEFAULT_TERMINATION_ARGS = {
    "n_max_gen": 100,
}

# Enable for verbose
PYMOO_PROGRESS = True
PYMOO_VERBOSE = False

# Disable for memory, enable for more data
PYMOO_HISTORY = False

def init_solver(
    eval_cb,
    n_var,
    n_obj,
    vtype=bool,
    xl=0,
    xu=1,
    custom_operators=None,
    seed=DEFAULT_SEED
):
    solver_cls = DEFAULT_MOO_SOLVER if n_obj > 1 else DEFAULT_SOO_SOLVER
    solver_operators = DEFAULT_BOOL_OPERATORS if vtype is bool else DEFAULT_FLOAT_OPERATORS

    if custom_operators:
        solver_operators.update(custom_operators)

    solver_params = {**DEFAULT_SOLVER_PARAMS, **solver_operators}

    solver = solver_cls(**solver_params)
    termination = DEFAULT_TERMINATION(**DEFAULT_TERMINATION_ARGS)

    solver.setup(
        ModelEvaluationProblem(
            eval_cb, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=vtype
        ),
        termination=termination,
        seed=seed,
        verbose=PYMOO_VERBOSE,
        save_history=PYMOO_HISTORY,
        progress=PYMOO_PROGRESS,
    )

    return solver
