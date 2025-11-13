import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone

from ..moo.solver.default_solver import init_solver
from ..fitting import HoldOutFitting
from ..metrics import BinaryConfusionMatrix

from .constrained import get_constrained_operators

DEFAULT_ESTIMATOR = GaussianNB()
DEFAULT_FITTING = HoldOutFitting()
DEFAULT_METRICS = ["TPR", "TNR"]

CONSTRAINED = True
USE_PROBA = False
STORE_SOLVER = True

class MOWS:
    def __init__(
        self,
        estimator=DEFAULT_ESTIMATOR,
        fitting=DEFAULT_FITTING,
        metrics=DEFAULT_METRICS,
        constrained=False,
        use_proba=False,
        step_callback=None
    ):
        self.estimator = estimator
        self.fitting = fitting
        self.metrics = metrics
        self.constrained = constrained
        self.use_proba = use_proba
        self.step_callback = step_callback

        self._solutions = []
        self._indicators = []
        self._models = []

    def fit(self, X, y):
        # Separate samples if needed i.e. for holdout
        X_train, y_train = self.fitting.initialize(X, y)
        self._classes, cc = np.unique(y_train, return_counts=True)

        if self.constrained:
            self.budget = np.max(cc)
            self.train_len = len(X_train)
            self.minority_indices = np.argwhere(y_train == self._classes.take(np.argmin(cc)))

            solver = init_solver(
                self._make_eval_cb(X_train, y_train),
                n_var=len(self.minority_indices),
                n_obj=len(self.metrics),
                custom_opeartors = get_constrained_operators(),
                vtype=float,
            )
        else:
            self.budget = np.sum(cc)
            solver = init_solver(
                self._make_eval_cb(X_train, y_train),
                n_var=len(X_train),
                n_obj=len(self.metrics),
                vtype=float,
            )

        while solver.has_next():
            solver.next()

            if self.step_callback:
                self.step_callback(self, solver)

        for s, i in zip(*solver.opt.get("X", "F")):
            self._solutions.append(s)
            self._indicators.append(i)

            clf = clone(self.estimator)
            weights = self.dispatch_solution(s)
            clf.fit(X_train, y_train, sample_weight=weights)

            self._models.append(clf)

        if STORE_SOLVER:
            self.solver = solver

        return self

    def predict(self, X):
        _ret = [c.predict(X) for c in self._models]
        return np.array(_ret)

    def dispatch_solution(self, s):
        if self.constrained:
            weights = np.ones(self.train_len, dtype=float).flatten()
            weights[self.minority_indices]
            return weights * self.budget
        else:
            return s * self.budget

    def _make_eval_cb(self, X, y):
        def _eval(S):
            scores, models = [], []

            for train, X_test, y_test in self.fitting:
                test_scores, test_models = [], []

                for s in S:
                    clf = clone(self.estimator)
                    weights = self.dispatch_solution(s)[train]
                    clf.fit(X[train], y[train], sample_weight=weights)

                    test_models.append(clf)

                    y_pred = clf.predict(X_test)
                    cm = BinaryConfusionMatrix(y_test, y_pred)
                    model_scores = cm.get_metrics(*self.metrics)

                    test_scores.append(model_scores)

                models.append(test_models)
                scores.append(test_scores)

            scores = np.mean(scores, axis=0)
            return models, scores

        return _eval
