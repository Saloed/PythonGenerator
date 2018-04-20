from typing import Iterable

import numpy as np

EPSILON = 1e-10


class Score:
    def __init__(self, TP: int, TN: int, FP: int, FN: int):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.Ps = self.TP + self.FP
        self.Ns = self.TN + self.FN
        self.T = self.TP + self.TN
        self.F = self.FP + self.FN
        self.P = self.TP + self.FN
        self.N = self.TN + self.FP
        self.ALL = self.P + self.N
        self.TPR = self.TP / (self.P + EPSILON)
        self.TNR = self.TN / (self.N + EPSILON)
        self.PPV = self.TP / (self.Ps + EPSILON)
        self.NPV = self.TN / (self.Ns + EPSILON)
        self.FNR = 1 - self.TPR
        self.FPR = 1 - self.TNR
        self.FDR = 1 - self.PPV
        self.FOR = 1 - self.NPV
        self.ACC = self.T / self.ALL
        self.MCC = (self.TP * self.TN - self.FP * self.FN) / (np.sqrt(self.Ps * self.P * self.N * self.Ns) + EPSILON)
        self.BM = self.TPR + self.TNR - 1
        self.MK = self.PPV + self.NPV - 1
        self.J = self.TP / (self.TP + self.FP + self.FN + EPSILON)
        self.recall = self.TPR
        self.precision = self.PPV
        self.accuracy = self.ACC
        self.true_positive = self.TP
        self.true_negative = self.TN
        self.false_negative = self.FN
        self.false_positive = self.FP
        self.jaccard = self.J

    def F_score(self, beta):
        beta = beta * beta
        return (1 + beta) * self.PPV * self.TPR / (beta * self.PPV + self.TPR + EPSILON)

    def E_score(self, alpha):
        return 1 - self.PPV * self.TPR / (alpha * self.TPR + (1 - alpha) * self.PPV + EPSILON)

    def serialize(self) -> dict:
        json_object = {
            "true_positive": self.TP,
            "true_negative": self.TN,
            "false_negative": self.FN,
            "false_positive": self.FP}
        return json_object

    @staticmethod
    def value_of(json_object: dict) -> 'Score':
        tp = int(json_object["true_positive"])
        tn = int(json_object["true_negative"])
        fp = int(json_object["false_negative"])
        fn = int(json_object["false_positive"])
        return Score(tp, tn, fp, fn)

    @staticmethod
    def calc(target, output, ignore, pad):
        target = np.asarray(target)
        output = np.asarray(output)
        tn = np.count_nonzero(np.logical_and(target == pad, output == pad))
        tp = np.count_nonzero(np.logical_and(np.logical_and(target == output, output != pad), target != pad))
        fp = np.count_nonzero(np.logical_and(np.logical_and(output != target, output != pad), target != ignore))
        fn = np.count_nonzero(np.logical_and(np.logical_and(output != target, target != pad), target != ignore))
        return Score(tp, tn, fp, fn)

    @staticmethod
    def concat(scores: Iterable['Score']) -> 'Score':
        tp = sum(score.TP for score in scores)
        tn = sum(score.TN for score in scores)
        fp = sum(score.FP for score in scores)
        fn = sum(score.FN for score in scores)
        return Score(tp, tn, fp, fn)
