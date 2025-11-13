import numpy as np
from functools import cached_property


def confusion_matrix(y_true, y_pred, n_classes=2):
    y = n_classes * y_true + y_pred
    y = np.bincount(y, minlength=(n_classes * n_classes))
    return y.reshape(n_classes, n_classes)


def error_confusion_matrix(y_true, y_proba):
    n_classes = len(np.unique(y_true))

    if len(y_proba.shape) == 1:
        raise NotImplementedError(
            "One-Class are not supported by error confusion matrix."
        )

    return np.vstack([np.sum(y_proba[y_true == l], axis=0) for l in range(n_classes)])


def zero_division(a, b):
    return np.divide(a, b, out=np.zeros(a.shape), where=b != 0)


class BinaryConfusionMatrix:
    def __init__(self, y_true, y_pred):
        if y_pred.dtype == int:
            if len(y_pred.shape) > 1:
                self.CM = np.array([confusion_matrix(y_true, p) for p in y_pred])
            else:
                self.CM = confusion_matrix(y_true, y_pred)
        else:
            self.CM = error_confusion_matrix(y_true, y_pred)

    @staticmethod
    def make_metrics_handler(*metrics):
        return lambda y_true, y_pred: BinaryConfusionMatrix(y_true, y_pred).get_metric(
            *metrics
        )

    # True Positive
    @cached_property
    def TP(self):
        return self.CM[..., 1, 1]

    # False Negative
    @cached_property
    def FN(self):
        return self.CM[..., 1, 0]

    # False Positive
    @cached_property
    def FP(self):
        return self.CM[..., 0, 1]

    # True Negative
    @cached_property
    def TN(self):
        return self.CM[..., 0, 0]

    # Actual Positive
    @cached_property
    def AP(self):
        return self.TP + self.FN

    # Actual Negative
    @cached_property
    def AN(self):
        return self.TN + self.FP

    # Predicted Positive
    @cached_property
    def PP(self):
        return self.TP + self.FP

    # Predicted Negative
    @cached_property
    def PN(self):
        return self.FN + self.TN

    # True Positive Rate
    @cached_property
    def TPR(self):
        # return self.TP / self.AP
        return zero_division(self.TP, self.AP)

    # False Negative Rate
    @cached_property
    def FNR(self):
        return 1 - self.TPR

    # True Negative Rate
    @cached_property
    def TNR(self):
        # return self.TN / self.AN
        return zero_division(self.TN, self.AN)

    # False Positive Rate
    @cached_property
    def FPR(self):
        return 1 - self.TNR

    # Positive Predictive Value
    @cached_property
    def PPV(self):
        # return self.TP / self.PP
        return zero_division(self.TP, self.PP)

    # False Discovery Rate
    @cached_property
    def FDR(self):
        return 1 - self.PPV

    # Negative Predictive Value
    @cached_property
    def NPV(self):
        # return self.TN / self.PN
        return zero_division(self.TN, self.PN)

    # False Ommision Rate
    @cached_property
    def FOR(self):
        return 1 - self.NPV

    # Positive Likehood Ratio
    @cached_property
    def PLR(self):
        # return self.TPR / self.FPR
        return zero_division(self.TPR, self.FPR)

    # Negative Likehood Ratio
    @cached_property
    def NLR(self):
        # return self.FNR / self.TNR
        return zero_division(self.FNR, self.TNR)

    # Diagnostic Odds Ratio
    @cached_property
    def DOR(self):
        # return self.PLR / self.NLR
        return zero_division(self.PLR, self.NLR)

    @cached_property
    def total_population(self):
        return self.AP + self.AN

    @cached_property
    def prevalence(self):
        return self.AP / self.total_population

    @cached_property
    def markednes(self):
        return self.PPV + self.NPV - 1

    @cached_property
    def accuracy(self):
        return (self.TP + self.TN) / self.total_population

    @cached_property
    def balanced_accuracy(self):
        return (self.TPR + self.TNR) / 2

    @cached_property
    def f1_score(self):
        return (2 * self.PPV * self.TPR) / (self.PPV + self.TPR)

    @cached_property
    def gmean(self):
        return np.sqrt(self.TPR * self.TNR)

    @cached_property
    def fowlkes_mallows_index(self):
        return np.sqrt(self.PPV * self.TPR)

    @cached_property
    def matthews_correlation_coefficient(self):
        return np.sqrt(
            (self.TPR * self.TNR * self.PPV * self.NPV)
            - (self.FNR * self.FPR * self.FOR * self.FDR)
        )

    @cached_property
    def jaccard_index(self):
        return self.TP / (self.TP + self.FN + self.FP)

    # Legacy definitions
    @property
    def sensitivity(self):
        return self.TPR

    @property
    def recall(self):
        return self.TPR

    @property
    def specificity(self):
        return self.TNR

    @property
    def precision(self):
        return self.PPV

    def f_beta_score(self, beta=1):
        beta_sqr = beta ^ 2
        return (1 + beta_sqr) * self.PPV * self.TPR / ((beta_sqr * self.PPV) + self.TPR)

    # Calculate multiple metrics
    def get_metrics(self, *metrics):
        return np.array([getattr(self, metric) for metric in metrics])

    def __str__(self):
        return str(self.CM)
