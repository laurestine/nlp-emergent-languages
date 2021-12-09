import numpy as np

from typing import List, Callable

from scipy.stats import spearmanr

"""
Adapted from
https://github.com/tomekkorbak/measuring-non-trivial-compositionality
"""


class TopographicSimilarity():

    def __init__(self, message_metric: Callable, representation_metric: Callable):
        self.message_metric = message_metric
        self.representation_metric = representation_metric

    def measure(self, compositional_representation_A, messages_A, compositional_representation_B, messages_B):
        distance_messages = self._compute_distances(
            sequence_A=messages_A,
            sequence_B=messages_B,
            metric=self.message_metric)

        distance_representation = self._compute_distances(
            sequence_A=compositional_representation_A,
            sequence_B=compositional_representation_B,
            metric=self.representation_metric)

        topsim = spearmanr(distance_representation,
                           distance_messages, nan_policy="raise").correlation

        return topsim

    def _compute_distances(self, sequence_A: List[str], sequence_B: List[str], metric: Callable) -> List[float]:
        distances = []
        for element_1 in sequence_A:
            for element_2 in sequence_B:
                distances.append(metric(element_1, element_2))
        return distances
