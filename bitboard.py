import color as c
import numpy as np

BLACK_BITS = np.uint64(0x0000000810000000)
WHITE_BITS = np.uint64(0x0000001008000000)


class BitBoard:
    def __init__(self, color: int, bits: np.uint64 = 0):
        self.color = color
        self.bits = bits if bits != 0 \
            else BLACK_BITS if color == c.BLACK \
            else WHITE_BITS

    def __str__(self):
        col_str = 'BLACK' if self.color == c.BLACK else 'WHITE'
        return f'Bitboard ({col_str}): [bin: {format(self.bits, "064b")} | decimal: {self.bits}]'

    def apply_isolated_move(self, move) -> None:
        mask = np.uint64(1 << move.pos)
        self.bits |= mask

    def get_cell_state(self, pos: int) -> bool:
        mask = np.uint64(1 << pos)
        return (self.bits & mask) != 0

    def get_bit_count(self):
        return np.binary_repr(self.bits).count('1')
