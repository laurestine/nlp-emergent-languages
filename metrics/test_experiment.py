import editdistance  # Levenshtein distance

from scipy.spatial.distance import hamming

from utils import transform_corpus
from topographic_similarity import TopographicSimilarity
from positional_disentanglement import PositionalDisentanglement

"""
Data
"""
input_A = ['blue circle', 'silver box', 'green circle']
compositional_representation_A = ['a!x_', 'jx__', 'c!x_']

input_B = ['blue box', 'red circle', 'green box']
compositional_representation_B = ['ax__', 'b!x_', 'cx__']

vocabulary = input_A + input_B

_, messages_A = transform_corpus(input_A, vocabulary)
print("Vocabulary: {} \n".format(_))
print("Count vector of A: \n {} \n".format(messages_A))

_, messages_B = transform_corpus(input_B, vocabulary)
print("Count vector of B: \n {} \n".format(messages_B))

"""
TopSim
"""
topsim_class = TopographicSimilarity(message_metric=hamming,
                                     representation_metric=editdistance.eval)
topsim = topsim_class.measure(compositional_representation_A, messages_A,
                              compositional_representation_B,  messages_B)

print("Topsim: {} \n".format(topsim))

"""
Pos
"""
pos_class = PositionalDisentanglement(
    max_message_length=4, num_concept_slots=2)
pos = pos_class.measure(compositional_representation_A, input_A)
print("Pos: {}".format(pos))
