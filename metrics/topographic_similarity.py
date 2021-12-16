import numpy as np

from typing import List, Callable

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from utils import transform_corpus

"""
Adapted from
https://github.com/tomekkorbak/measuring-non-trivial-compositionality
https://github.com/facebookresearch/EGG/blob/2d2ec2a19fa20502494e2b49ba8d8d2fd4036734/egg/core/language_analysis.py#L124
"""


class TopographicSimilarity():
    def __init__(self, message_metric: Callable, meaning_metric: Callable):
        self.message_metric = message_metric
        self.meaning_metric = meaning_metric

    def measure(self, meanings: List[str], messages: List[str]) -> float:
        mean = [[ord(c) for c in w] for w in meanings]
        mess = transform_corpus(messages)

        distance_messages = pdist(mess,self.message_metric)
        distance_meaning = pdist(mean,self.meaning_metric)

        topsim = spearmanr(distance_meaning,
                           distance_messages, nan_policy="raise").correlation

        return topsim
