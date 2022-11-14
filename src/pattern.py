import itertools
import logging
import os

import numpy as np

from src.block import generate_full_pattern_matrix_in_blocks
from src.file import get_pattern_matrix_fname
from src.pattern_utils import EXACT, MISPLACED, MISS, generate_pattern_matrix
from src.prior import get_word_list

# To store the large grid of patterns at run time
PATTERN_GRID_DATA = {}


# Generating color patterns between strings, etc.


def generate_full_pattern_matrix(game_name):
    words = get_word_list(game_name)
    pattern_matrix = generate_full_pattern_matrix_in_blocks(words)
    # Save to file
    np.save(get_pattern_matrix_fname(game_name), pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2, game_name):
    pattern_matrix_fname = get_pattern_matrix_fname(game_name)
    if not PATTERN_GRID_DATA:
        if not os.path.exists(pattern_matrix_fname):
            logging.info(
                "\n".join(
                    [
                        "Generating pattern matrix. This takes a minute, but",
                        "the result will be saved to file so that it only",
                        "needs to be computed once.",
                    ],
                ),
            )
            generate_full_pattern_matrix(game_name)
        PATTERN_GRID_DATA["grid"] = np.load(pattern_matrix_fname)
        PATTERN_GRID_DATA["words_to_index"] = dict(
            zip(get_word_list(game_name), itertools.count()),
        )

    full_grid = PATTERN_GRID_DATA["grid"]
    words_to_index = PATTERN_GRID_DATA["words_to_index"]

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def get_pattern(guess, answer, game_name):
    if PATTERN_GRID_DATA:
        saved_words = PATTERN_GRID_DATA["words_to_index"]
        if guess in saved_words and answer in saved_words:
            return get_pattern_matrix([guess], [answer], game_name)[0, 0]
    return generate_pattern_matrix([guess], [answer])[0, 0]


def pattern_to_int_list(pattern):
    result = []
    curr = pattern
    for _x in range(5):
        result.append(curr % 3)
        curr = curr // 3
    return result


def pattern_to_string(pattern):
    d = {MISS: "⬛", MISPLACED: "🟨", EXACT: "🟩"}
    return "".join(d[x] for x in pattern_to_int_list(pattern))


def patterns_to_string(patterns):
    return "\n".join(map(pattern_to_string, patterns))


def get_possible_words(guess, pattern, word_list, game_name):
    all_patterns = get_pattern_matrix([guess], word_list, game_name).flatten()
    return list(np.array(word_list)[all_patterns == pattern])


def get_word_buckets(guess, possible_words, game_name):
    buckets = [[] for _x in range(3**5)]
    hashes = get_pattern_matrix([guess], possible_words, game_name).flatten()
    for index, word in zip(hashes, possible_words):
        buckets[index].append(word)
    return buckets
