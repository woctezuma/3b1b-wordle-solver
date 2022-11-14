import numpy as np

from src.pattern_utils import generate_pattern_matrix

CHUNK_SIZE = 13000


def chunks(lst, length):
    """Yield successive n-sized chunks from lst.
    Reference: https://stackoverflow.com/a/312464/376454
    """
    for i in range(0, len(lst), length):
        yield lst[i : i + length]


def generate_full_pattern_matrix_in_blocks(words, length=CHUNK_SIZE):
    block_matrix = None
    for words1 in chunks(words, length):
        row = None

        for words2 in chunks(words, length):
            block = generate_pattern_matrix(words1, words2)

            if row is None:
                row = block
            else:
                row = np.hstack((row, block))

        if block_matrix is None:
            block_matrix = row
        else:
            block_matrix = np.vstack((block_matrix, row))

    return block_matrix
