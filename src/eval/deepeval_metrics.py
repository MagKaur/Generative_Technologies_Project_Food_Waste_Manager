# src/eval/deepeval_metrics.py
from dataclasses import dataclass
from typing import List, Set
import math

from deepeval.metrics import BaseMetric


@dataclass
class RetrievalCase:
    qid: str
    query: str
    ranked: List[str]       # ranking recipe_uuid
    relevant: Set[str]      # GT set recipe_uuid
    k: int


class PrecisionAtKMetric(BaseMetric):
    def __init__(self, k: int):
        self.k = k
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: RetrievalCase) -> float:
        top = test_case.ranked[: self.k]
        self.score = (sum(1 for x in top if x in test_case.relevant) / len(top)) if top else 0.0
        self.reason = f"P@{self.k}"
        return self.score

    def is_successful(self) -> bool:
        return True


class RecallAtKMetric(BaseMetric):
    def __init__(self, k: int):
        self.k = k
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: RetrievalCase) -> float:
        if not test_case.relevant:
            self.score = 0.0
            self.reason = "Empty GT"
            return self.score
        top = test_case.ranked[: self.k]
        self.score = sum(1 for x in top if x in test_case.relevant) / len(test_case.relevant)
        self.reason = f"R@{self.k}"
        return self.score

    def is_successful(self) -> bool:
        return True


class MRRAtKMetric(BaseMetric):
    def __init__(self, k: int):
        self.k = k
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: RetrievalCase) -> float:
        top = test_case.ranked[: self.k]
        for i, x in enumerate(top, start=1):
            if x in test_case.relevant:
                self.score = 1.0 / i
                self.reason = f"First relevant @ rank {i}"
                return self.score
        self.score = 0.0
        self.reason = "No relevant in top-k"
        return self.score

    def is_successful(self) -> bool:
        return True


class NDCGAtKMetric(BaseMetric):
    def __init__(self, k: int):
        self.k = k
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: RetrievalCase) -> float:
        top = test_case.ranked[: self.k]

        def dcg(ids: List[str]) -> float:
            s = 0.0
            for i, x in enumerate(ids, start=1):
                rel = 1.0 if x in test_case.relevant else 0.0
                s += rel / math.log2(i + 1)
            return s

        dcg_val = dcg(top)
        ideal_rel = min(self.k, len(test_case.relevant))
        ideal = sum((1.0 / math.log2(i + 1)) for i in range(1, ideal_rel + 1))
        self.score = (dcg_val / ideal) if ideal > 0 else 0.0
        self.reason = f"nDCG@{self.k}"
        return self.score

    def is_successful(self) -> bool:
        return True
