import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone

from ..moo.solver.default_solver import init_solver
from ..fitting import DummyFitting
from ..metrics import BinaryConfusionMatrix
from ..moo.mcdm.saw import saw
from copy import copy

from .constrained import get_constrained_operators

DEFAULT_ESTIMATOR = GaussianNB()
DEFAULT_FITTING = DummyFitting()
DEFAULT_METRICS = ["TPR", "TNR"]

class MOWS:
    def __init__(
        self,
        estimator=DEFAULT_ESTIMATOR,
        fitting=DEFAULT_FITTING,
        metrics=DEFAULT_METRICS,
        constrained=False,
        use_proba=False,
        verbose=False,
        random_state=None,
    ):
        self.estimator = clone(estimator)
        self.fitting = copy(fitting)
        self.metrics = metrics
        self.constrained = constrained
        self.use_proba = use_proba
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, step_cb=None, store_solver=False):
        # Separate samples if needed i.e. for holdout
        self.X_train, self.y_train = self.fitting.initialize(X, y)
        self._classes, cc = np.unique(self.y_train, return_counts=True)

        if self.constrained:
            self.budget = np.max(cc)
            self.train_len = len(self.X_train)
            self.minority_indices = np.argwhere(self.y_train == self._classes.take(np.argmin(cc)))

            solver = init_solver(
                self._make_eval_cb(),
                n_var=len(self.minority_indices),
                n_obj=len(self.metrics),
                custom_operators = get_constrained_operators(),
                vtype=float,
                verbose=self.verbose,
                seed=self.random_state
            )
        else:
            self.budget = np.sum(cc)
            solver = init_solver(
                self._make_eval_cb(),
                n_var=len(self.X_train),
                n_obj=len(self.metrics),
                vtype=float,
                verbose=self.verbose,
                seed=self.random_state
            )

        while solver.has_next():
            solver.next()

            if step_cb:
                step_cb(self, solver)

        self._solutions = solver.opt.get("X")
        self._indicators = solver.opt.get("F")
        self._models = solver.opt.get("A")

        if self._models.shape[1] > 1:
            # Adjust to CV models
            self._models = np.array([self.train_estimator(s) for s in self._solutions])

        self._models = self._models.ravel()

        if store_solver:
            self._solver = solver

        return self

    def predict(self, X, pref=None):
        return self._models[saw(self._indicators, pref)].predict(X) if pref else np.array([m.predict(X) for m in self._models])

    def predict_proba(self, X, pref=None):
        return self._models[saw(self._indicators, pref)].predict_proba(X) if pref else np.array([m.predict_proba(X) for m in self._models])


    def translate_solution(self, s):
        if self.constrained:
            weights = np.ones(self.train_len, dtype=float).ravel()
            weights[self.minority_indices] = (s * self.budget)[:, None]
            return weights
        else:
            return s * self.budget

    def train_estimator(self, s, training_map=None):
        if training_map is None:
            training_map = np.arange(len(self.X_train))

        weights = self.translate_solution(s)[training_map]
        clf = clone(self.estimator)
        clf.fit(self.X_train[training_map], self.y_train[training_map], sample_weight=weights)

        return clf

    def _make_eval_cb(self):
        def _eval(S):
            scores, models = [], []

            for train, X_test, y_test in self.fitting:
                test_scores = []
                test_models = []

                for s in S:
                    # Train model
                    clf = self.train_estimator(s, train)
                    # Evaluate model
                    y_pred = clf.predict_proba(X_test) if self.use_proba else clf.predict(X_test)
                    model_scores = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*self.metrics)

                    # Store artifacts
                    test_scores.append(model_scores)
                    test_models.append(clf)

                scores.append(test_scores)
                models.append(test_models)

            # Prepare for storing
            scores = -1 * np.mean(scores, axis=0)
            models = np.array(models).T

            return scores, models
        return _eval
