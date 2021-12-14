import unittest
import gameboard
import bitboard
import color
import numpy as np
import output
from move import Move


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

    @staticmethod
    def test_starter_move_gen(possible, board):
        generated = board.generate_moves_priority_queue(board.player_board, board.opponent_board)
        while not generated.empty():
            p_item = generated.get()
            if p_item.move.pos not in possible:
                return False, p_item

        return True, None


class TestOthello(unittest.TestCase):
    BLACK_BITS = np.uint64(0x0000000810000000)
    WHITE_BITS = np.uint64(0x0000001008000000)

    def test_starting_configuration_black(self):
        board = Constants.default_board_black()
        self.assertEqual(board.player_board.bits, self.BLACK_BITS, "Black starting bits do not match player.")

    def test_starting_configuration_white(self):
        board = Constants.default_board_white()
        self.assertEqual(board.player_board.bits, self.WHITE_BITS, "White starting bits do not match player.")

    def test_starter_move_gen_black(self):
        # Generating moves for black
        possible = [19, 26, 37, 44]
        board = Constants.default_board_black()
        res = Constants.test_starter_move_gen(possible, board)
        self.assertTrue(res[0], f'Invalid move generated. Possible choices: {str(possible)}, '
                                f'received {res[1].move if res[1] is not None else None}')
        # Silly, but the inline if has to be there to avoid errors. Even on successful tests.

    def test_starter_move_gen_white(self):
        # Generating moves for white
        possible = [20, 29, 34, 43]
        board = Constants.default_board_white()
        res = Constants.test_starter_move_gen(possible, board)
        self.assertTrue(res[0], f'Invalid move generated. Possible choices: {str(possible)}, '
                                f'received {res[1].move if res[1] is not None else None}')

    def test_select_random_move(self):
        board = Constants.default_board_black()
        queue = board.generate_moves_priority_queue(board.player_board, board.opponent_board)
        random = board.select_random_move(board.player_color)
        valid = random.pos in [x.move.pos for x in queue.items]
        self.assertTrue(valid, f'Expected to find random move {random} in queue\n{queue}but did not.')

    def test_pass_sequence(self):
        """Tests a situation where no legal moves can be generated for a player. Move generated must be a pass."""
        board = Constants.default_board_black()
        move_sequence = ['B c 4', 'W e 3', 'B f 3', 'W g 3', 'B g 2', 'W c 5', 'B c 6', 'W b 5', 'B e 6', 'W d 7',
                         'B c 3', 'W h 1', 'B c 8', 'W f 7', 'B a 6', 'W d 3', 'B h 3', 'W f 2', 'B f 1', 'W g 1',
                         'B e 1', 'W d 1', 'B d 6', 'W h 4', 'B h 5', 'W h 2', 'B f 6', 'W h 6', 'B f 8', 'W g 6',
                         'B f 5', 'W e 8', 'B d 8', 'W c 2', 'B f 4', 'W g 4', 'B b 4', 'W e 2', 'B c 1', 'W b 6',
                         'B d 2', 'W g 5', 'B c 7', 'W b 1', 'B b 7', 'W a 8', 'B b 8', 'W g 7', 'B h 8', 'W a 4',
                         'B b 3', 'W a 3', 'B e 7', 'W a 5', 'B a 2', 'W b 2', 'B a 1',
                         'W a 7']  # black must pass at this point
        for m in move_sequence:
            if m.startswith('W'):
                c = color.WHITE
            else:
                c = color.BLACK

            bitboard = board.get_for_color(c)
            move = output.to_move(m, False)
            board.apply_move(bitboard, move)

        # Board is now at a state where black has no available moves.
        possible = board.generate_moves_priority_queue(board.player_board, board.opponent_board)
        self.assertTrue(len(possible.items) == 1 and possible.items[0].move.pos == Move(0).pos,
                        f'Expected one pass move to be sole pass move.\n{board}\n{possible.items}')

    def test_input_conversion(self):
        chars = 'abcdefgh'
        nums = [x for x in range(1, 9)]
        colors = 'WB'

        fake_input = []
        validations = []
        for c in colors:
            for n in nums:
                for l in chars:
                    fake_input.append(f'{c} {l} {n}')

        for c in colors:
            c_num = color.WHITE if c == 'W' else color.BLACK
            z = 63
            for _ in range(len(chars) * len(nums)):
                validations.append(Move(c_num, z, is_pass=False))
                z -= 1

        valid_pairs = [(x, y) for x, y in zip(fake_input, validations)]
        test_pairs = []
        for i in fake_input:
            test_pairs.append((i, output.to_move(i, False)))

        for i in range(len(valid_pairs)):
            self.assertTrue(valid_pairs[i][1].pos == test_pairs[i][1].pos,
                            f'Failure on comparison. {valid_pairs[i][1].pos} did not equal {test_pairs[i][1].pos}')


if __name__ == '__main__':
    unittest.main()
