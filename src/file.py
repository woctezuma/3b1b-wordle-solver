import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
SHORT_WORD_LIST_FILE = "possible_words.txt"
LONG_WORD_LIST_FILE = "allowed_words.txt"
WORD_FREQ_FILE = "wordle_words_freq_full.txt"
WORD_FREQ_MAP_FILE = "freq_map.json"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
SIMULATION_DIR = "simulation_results"


def get_data_dir(game_name):
    return os.path.join(DATA_DIR, game_name)


def get_data_fname(game_name, file):
    return os.path.join(get_data_dir(game_name), file)


def get_short_word_list_fname(game_name):
    return get_data_fname(game_name, SHORT_WORD_LIST_FILE)


def get_long_word_list_fname(game_name):
    return get_data_fname(game_name, LONG_WORD_LIST_FILE)


def get_word_freq_fname(game_name):
    return get_data_fname(game_name, WORD_FREQ_FILE)


def get_word_freq_map_fname(game_name):
    return get_data_fname(game_name, WORD_FREQ_MAP_FILE)


def get_pattern_matrix_fname(game_name):
    return get_data_fname(game_name, PATTERN_MATRIX_FILE)


def get_simulation_results_folder(game_name):
    return get_data_fname(game_name, SIMULATION_DIR)
