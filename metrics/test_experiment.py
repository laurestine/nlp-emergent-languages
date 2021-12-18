import string
import random
import editdistance  # Levenshtein distance
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from scipy.spatial.distance import hamming

from topographic_similarity import TopographicSimilarity
from positional_disentanglement import PositionalDisentanglement
from conflict_count import ConflictCount

"""
Tests were performed based on the results of
https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/master/experiment_1.py
"""


"""
Data
"""

def get_negation_ntc_protocol():
    '''
    From https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/2b365626ad256c94dbd8d7eb041ef98b1028796a/protocols.py#L86
    '''
    # '!' = negation, 'x' = box, 'a' = blue, 'b' = red, ..., '_' = padding token
    return {
        ('blue', 'circle'): 'a!x_',  # blue not box
        ('blue', 'box'): 'ax__',  # blue box

        ('red', 'circle'): 'b!x_',  # red not box
        ('red', 'box'): 'bx__',  # red box

        ('green', 'circle'): 'c!x_',  # green not box
        ('green', 'box'): 'cx__',  # green box

        ('yellow', 'circle'): 'd!x_',  # yellow not box_
        ('yellow', 'box'): 'dx__',  # yellow box

        ('gold', 'circle'): 'e!x_',  # gold not box
        ('gold', 'box'): 'ex__',  # gold box

        ('orange', 'circle'): 'f!x_',  # orange not box
        ('orange', 'box'): 'fx__',  # orange box

        ('white', 'circle'): 'g!x_',  # white not box
        ('white', 'box'): 'gx__',  # white box

        ('black', 'circle'): 'h!x_',  # black not box
        ('black', 'box'): 'hx__',  # black box

        ('pink', 'circle'): 'i!x_',  # pink not box
        ('pink', 'box'): 'ix__',  # pink box

        ('silver', 'circle'): 'j!x_',  # silver not box
        ('silver', 'box'): 'jx__',  # silver box

        ('bronze', 'circle'): 'k!x_',  # bronze not box
        ('bronze', 'box'): 'kx__',  # bronze box
    }

negative_sentence_data = get_negation_ntc_protocol()

negative_sentence_input = []
negative_sentence_meaning = []

for label, value in negative_sentence_data.items():
    aux_list = list(label)
    negative_sentence_input.append(aux_list[0]+' '+aux_list[1])
    negative_sentence_meaning.append(value)


POSSIBLE_COLORS = ['blue', 'green', 'gold', 'yellow', 'red', 'orange'] + [f'color_{i}' for i in range(25)]
POSSIBLE_SHAPES = ['square', 'circle', 'ellipse', 'triangle', 'rectangle', 'pentagon'] + [f'shape_{i}' for i in range(25)]

def get_rotated_ntc_protocol(num_colors: int, num_shapes: int):
    """
    From
    https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/2b365626ad256c94dbd8d7eb041ef98b1028796a/protocols.py#L152
    """
    num_letters = num_colors + num_shapes
    alphabet = list(string.ascii_letters[:num_letters])
    mapping = {}
    random.shuffle(alphabet)
    for i, color in enumerate(POSSIBLE_COLORS[:num_colors]):
        for j, shape in enumerate(POSSIBLE_SHAPES[:num_shapes]):
            first_letter = alphabet[i - j + num_shapes]
            second_letter = alphabet[j + i]
            mapping[color, shape] = first_letter + second_letter
    return mapping

NUM_COLORS = NUM_SHAPES = 25
rotation_sentence_data = get_rotated_ntc_protocol(NUM_COLORS, NUM_SHAPES)

rotation_sentence_input = []
rotation_sentence_meaning = []

for label, value in rotation_sentence_data.items():
    aux_list = list(label)
    rotation_sentence_input.append(aux_list[0]+' '+aux_list[1])
    rotation_sentence_meaning.append(value)


"""
TopSim
"""

topsim_class = TopographicSimilarity(message_metric=hamming,
                                     meaning_metric=editdistance.eval)
negative_sentence_meaning_ = [[ord(c) for c in w] for w in negative_sentence_meaning]
topsim_neg = topsim_class.measure(negative_sentence_meaning_, negative_sentence_input)

print("Topsim - negation : {} \n".format(topsim_neg)) # 0.9770084209183943

rotation_sentence_meaning_ = [[ord(c) for c in w] for w in rotation_sentence_meaning]
topsim_rot = topsim_class.measure(rotation_sentence_meaning_, rotation_sentence_input)
print("Topsim - rotation : {} \n".format(topsim_rot)) # -0.06640517470966958


"""
Pos
"""
pos_class = PositionalDisentanglement(max_message_length=4, num_concept_slots=2)
token_messages = [word_tokenize(sentence) for sentence in negative_sentence_input]
pos = pos_class.measure(negative_sentence_meaning, token_messages)
print("Pos: {} \n".format(pos)) # 0.999999999999999


"""
Conflict Count
"""

conf_count_class = ConflictCount(max_length=2)
conf_count = conf_count_class.measure(rotation_sentence_meaning, rotation_sentence_input)
print("Conflict count: {}".format(conf_count)) # 1152