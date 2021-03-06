import numpy as np

from typing import List

from utils import compute_entropy, compute_mutual_information


"""
Adapted from
https://github.com/tomekkorbak/measuring-non-trivial-compositionality
"""


class PositionalDisentanglement():
    def __init__(self, max_message_length: int, num_concept_slots: int):
        self.max_message_length = max_message_length
        self.num_concept_slots = num_concept_slots

    def measure(self, meanings: List[str], token_messages: List[List[str]]) -> float:
        disentanglement_scores = []
        non_constant_positions = 0

        for j in range(self.max_message_length):
            symbols_j = [mean[j] for mean in meanings]
            symbol_mutual_info = []
            symbol_entropy = compute_entropy(symbols_j)
            for i in range(self.num_concept_slots):
                concepts_i = [derivation[i] for derivation in token_messages]
                mutual_info = compute_mutual_information(concepts_i, symbols_j)
                symbol_mutual_info.append(mutual_info)
            symbol_mutual_info.sort(reverse=True)

            if symbol_entropy > 0:
                disentanglement_score = (symbol_mutual_info[0] - symbol_mutual_info[1]) / symbol_entropy
                disentanglement_scores.append(disentanglement_score)
                non_constant_positions += 1
        if non_constant_positions > 0:
            return sum(disentanglement_scores)/non_constant_positions
        else:
            return np.nan
