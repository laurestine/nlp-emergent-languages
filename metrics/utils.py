import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from typing import List


def transform_corpus(corpus: List[str], vocabulary: List[str] = []):
    vectorizer = CountVectorizer()
    if not vocabulary:
        matrix = vectorizer.fit_transform(corpus)
    else:
        vectorizer.fit(vocabulary)
        matrix = vectorizer.transform(corpus)

    return matrix.toarray()


def get_meaning(data,colunm: str, vocabulary: List[str] = []):
    meaning_col = data[colunm].copy()
    meaning_separeted = meaning_col.apply(lambda x: x.split(';'))
    
    if not vocabulary:
        all_meanings = [item for sublist in meaning_separeted for item in sublist]
        vocabulary = list(set(all_meanings))
    
    n_meanings = float('-inf')
    for meaning_list in meaning_separeted:
        n_meanings = n_meanings if n_meanings > len(meaning_list) else len(meaning_list)


    final_meanings = []
    for meaning_list in meaning_separeted:
        aux_meaning = np.zeros(n_meanings)
        for i in range(len(meaning_list)):
            aux_meaning[i] = vocabulary.index(meaning_list[i])+1
        final_meanings.append(aux_meaning)

    return final_meanings


def compute_entropy(symbols: List[str]) -> float:
    """
    From
    https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/2b365626ad256c94dbd8d7eb041ef98b1028796a/metrics/disentanglement.py#L17
    """
    frequency_table = defaultdict(float)
    for symbol in symbols:
        frequency_table[symbol] += 1.0
    H = 0
    for symbol in frequency_table:
        p = frequency_table[symbol]/len(symbols)
        H += -p * np.log2(p)
    return H

def compute_mutual_information(concepts: List[str], symbols: List[str]) -> float:
    """
    From
    https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/2b365626ad256c94dbd8d7eb041ef98b1028796a/metrics/disentanglement.py#L28
    """
    concept_entropy = compute_entropy(concepts)  # H[p(concepts)]
    symbol_entropy = compute_entropy(symbols)  # H[p(symbols)]
    symbols_and_concepts = [symbol + '_' + concept for symbol, concept in zip(symbols, concepts)]
    symbol_concept_joint_entropy = compute_entropy(symbols_and_concepts)  # H[p(concepts, symbols)]
    return concept_entropy + symbol_entropy - symbol_concept_joint_entropy