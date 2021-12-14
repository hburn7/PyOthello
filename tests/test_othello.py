import unittest
import gameboard
import bitboard
import color
import numpy as np

class Constants:
    @staticmethod
    def default_board_black():
        """Returns default gameboard with black as main player"""
        black = bitboard.BitBoard(color.BLACK)
        white = bitboard.BitBoard(color.WHITE)
        return gameboard.GameBoard(None, black, white)

    @staticmethod
    def default_board_white():
        """Returns default gameboard with white as main player"""
        black = bitboard.BitBoard(color.BLACK)
        white = bitboard.BitBoard(color.WHITE)
        return gameboard.GameBoard(None, white, black)

class TestOthello(unittest.TestCase):
    BLACK_BITS = np.uint64(0x0000000810000000)
    WHITE_BITS = np.uint64(0x0000001008000000)

    def test_starting_configuration_black(self):
        board = Constants.default_board_black()
        self.assertEqual(board.player_board.bits, self.BLACK_BITS, "Black starting bits do not match player.")

    def test_starting_configuration_white(self):
        board = Constants.default_board_white()
        self.assertEqual(board.player_board.bits, self.WHITE_BITS, "White starting bits do not match player.")

    def test_move_generation(self):
        

    def test_select_random_move(self):


        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
