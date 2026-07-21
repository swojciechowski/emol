import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone

from ..moo.solver.default_solver import init_solver
from ..moo.mcdm.saw import saw
from ..fitting import HoldOutFitting
from ..metrics import BinaryConfusionMatrix

from copy import copy

from .constrained import get_constrained_operators

DEFAULT_ESTIMATOR = GaussianNB()
DEFAULT_FITTING = HoldOutFitting()
DEFAULT_METRICS = ["TPR", "TNR"]

class MEUS:
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
            self.train_len = len(self.X_train)
            self.majority_indices = np.argwhere(self.y_train == self._classes.take(np.argmax(cc)))

        custom_operators = get_constrained_operators(n_max=np.min(cc)) if self.constrained else {}
        n_var = np.max(cc) if self.constrained else len(self.X_train)

        solver = init_solver(
            self._eval_cb(),
            n_var=n_var,
            n_obj=len(self.metrics),
            vtype=bool,
            custom_operators = custom_operators,
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
        return self._models[saw(self._indicators, pref)].predict(X) if pref is not None else np.array([m.predict(X) for m in self._models])

    def resample(self, s):
        us_map = self.translate_solution(s)
        return self.X_train[us_map], self.y_train[us_map]

    def train_estimator(self, s, training_map=None):
        if training_map is None:
            training_map = np.arange(len(self.X_train))

        us_map = self.translate_solution(s)[training_map]
        X_train, y_train = self.X_train[training_map][us_map], self.y_train[training_map][us_map]

        clf = clone(self.estimator)
        clf.fit(X_train, y_train)

        return clf

    def translate_solution(self, s):
        if self.constrained:
            selection_map = np.ones(self.train_len, dtype=bool)
            selection_map[self.majority_indices[np.logical_not(s)]] = 0
            return selection_map
        else:
            return s

    def _eval_cb(self):
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
