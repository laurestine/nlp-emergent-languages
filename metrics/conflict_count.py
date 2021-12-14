import itertools
import collections

from typing import List

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class ConflictCount():
    def __init__(self, max_length: int):
        self.max_length = max_length

    def measure(self, representations: List[str], messages: List[str]) -> float:
        all_conflicts = []
        token_messages = [word_tokenize(setence) for setence in messages]

        for p in itertools.permutations(range(self.max_length)):
            meanings = [collections.defaultdict(collections.Counter)
                        for i in range(self.max_length)]
            
            for idx,msg in enumerate(token_messages):
                for i in range(self.max_length):
                    meaning = representations[idx][i]
                    word = msg[p[i]]
                    meanings[i][meaning].update([word])
            
            conflicts = 0
            for meaning in meanings:
                for symbol in meaning.values():
                    conflicts += sum(v for c, v in symbol.most_common()[1:])
            all_conflicts += [conflicts]
        return min(all_conflicts)