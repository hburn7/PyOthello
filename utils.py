import numpy as np

# Inefficient?
import color
import move


def __dict():
    keys = 'abcdefgh'
    vals = [x for x in range(8)]
    return {k: v for k, v in zip(keys, vals)}


def count_bits(binary):
    """Returns the count of set bits in a given number."""
    return np.binary_repr(binary).count('1')


def col_from_char(col: str):
    return __dict().get(col)


def col_from_int(col: int):
    cols = 'abcdefgh'
    return cols[col]


def pos_to_row_col(pos) -> (int, str):
    """Returns a tuple containing the readable row integer and the readable column letter"""
    row = 7 - int((pos / 8))
    col = 7 - (pos % 8)

    if row < 0:
        row = -row
    if col < 0:
        col = -col

    conv_col = col_from_int(col)

    return row + 1, conv_col


def color_char(c: int) -> str:
    return 'W' if c == color.WHITE else 'B'

