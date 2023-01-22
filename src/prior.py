import json
import os

import numpy as np
from scipy.special import expit as sigmoid

from src.file import (
    get_long_word_list_fname,
    get_short_word_list_fname,
    get_word_freq_fname,
    get_word_freq_map_fname,
)

# Reading from files


def get_word_list(game_name, short=False):
    result = []
    file = (
        get_short_word_list_fname(game_name)
        if short
        else get_long_word_list_fname(game_name)
    )
    with open(file, encoding="utf8") as fp:
        result.extend([word.strip() for word in fp.readlines()])
    return result


def get_word_frequencies(game_name, regenerate=False):
    word_freq_map_fname = get_word_freq_map_fname(game_name)
    if os.path.exists(word_freq_map_fname) or regenerate:
        with open(word_freq_map_fname, encoding="utf8") as fp:
            result = json.load(fp)
        return result
    # Otherwise, regenerate
    freq_map = {}
    with open(get_word_freq_fname(game_name), encoding="utf8") as fp:
        for line in fp.readlines():
            pieces = line.split(" ")
            word = pieces[0]
            freq = [float(piece.strip()) for piece in pieces[1:]]
            freq_map[word] = np.mean(freq[-5:])
    with open(word_freq_map_fname, "w", encoding="utf8") as fp:
        json.dump(freq_map, fp)
    return freq_map


def get_frequency_based_priors(game_name, n_common=3000, width_under_sigmoid=10):
    """
    We know that that list of wordle answers was curated by some human
    based on whether they're sufficiently common. This function aims
    to associate each word with the likelihood that it would actually
    be selected for the final answer.

    Sort the words by frequency, then apply a sigmoid along it.
    """
    freq_map = get_word_frequencies(game_name)
    words = np.array(list(freq_map.keys()))
    freq = np.array([freq_map[w] for w in words])
    arg_sort = freq.argsort()
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = width_under_sigmoid
    c = x_width * (-0.5 + n_common / len(words))
    xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
    priors = {}
    for word, x in zip(sorted_words, xs, strict=True):
        priors[word] = sigmoid(x)
    return priors


def get_true_wordle_prior(game_name):
    words = get_word_list(game_name)
    short_words = get_word_list(game_name, short=True)
    return {w: int(w in short_words) for w in words}
