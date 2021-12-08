import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from typing import List

"""
Adapted from
https://github.com/tomekkorbak/measuring-non-trivial-compositionality
"""


def transform_corpus(corpus: List[str], vocabulary: List[str]):
    vectorizer = CountVectorizer()
    vectorizer.fit(vocabulary)
    matrix = vectorizer.transform(corpus)

    print('features', vectorizer.get_feature_names_out())

    return vectorizer.get_feature_names_out(), matrix.toarray()


def compute_entropy(symbols: List[str]) -> float:
    frequency_table = defaultdict(float)
    for symbol in symbols:
        frequency_table[symbol] += 1.0
    H = 0
    for symbol in frequency_table:
        p = frequency_table[symbol]/len(symbols)
        H += -p * np.log2(p)
    return H

def compute_mutual_information(concepts: List[str], symbols: List[str]) -> float:
    concept_entropy = compute_entropy(concepts)  # H[p(concepts)]
    symbol_entropy = compute_entropy(symbols)  # H[p(symbols)]
    symbols_and_concepts = [symbol + '_' + concept for symbol, concept in zip(symbols, concepts)]
    symbol_concept_joint_entropy = compute_entropy(symbols_and_concepts)  # H[p(concepts, symbols)]
    return concept_entropy + symbol_entropy - symbol_concept_joint_entropy