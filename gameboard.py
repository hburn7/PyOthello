import copy

import color
import move as m
import logger
import numpy as np
import utils
from datetime import datetime
from dataclasses import dataclass

from config import Config
from queue import PriorityQueue


@dataclass
class SearchResult:
    depth: int
    player: int
    score: int


class GameBoard:
    DIRECTION_COUNT = 8
    UNIVERSE = np.uint64(0xffffffffffffffff)

    CORNER_MASK = np.uint64(0x8100000000000081)
    CORNER_ADJACENT_MASK = np.uint64(0x42C3000000C342)

    DIR_INCREMENTS = np.array([8, 9, 1, -7, -8, -9, -1, 7], dtype=np.int32)
    DIR_MASKS = np.array([
        0xFFFFFFFFFFFFFF00,  # North
        0xFEFEFEFEFEFEFE00,  # NorthWest
        0xFEFEFEFEFEFEFEFE,  # West
        0x00FEFEFEFEFEFEFE,  # SouthWest
        0x00FFFFFFFFFFFFFF,  # South
        0x007F7F7F7F7F7F7F,  # SouthEast
        0x7F7F7F7F7F7F7F7F,  # East
        0x7F7F7F7F7F7F7F00  # NorthEast
    ], dtype=np.uint64)

    WEIGHT_MAP = np.array([
        50, -20, 11, 8, 8, 11, -20, 50,
        -20, -35, -4, 1, 1, -4, -35, -20,
        11, -4, 2, 2, 2, 2, -4, 11,
        8, 1, 2, 0, 0, 2, 1, 8,
        8, 1, 2, 0, 0, 2, 1, 8,
        11, -4, 2, 2, 2, 2, -4, 11,
        -20, -35, -4, 1, 1, -4, -35, -20,
        50, -20, 11, 8, 8, 11, -20, 50
    ])

    from bitboard import BitBoard
    def __init__(self, config: Config, player: BitBoard, opponent: BitBoard):
        self.config = config
        self.player_color = player.color
        self.opponent_color = -player.color
        self.player_board = player
        self.opponent_board = opponent

    def draw(self):
        logger.log_comment('    A B C D E F G H')
        logger.log_comment('    * * * * * * * *')

        black = self.get_for_color(color.BLACK)
        white = self.get_for_color(color.WHITE)

        for i in range(63, -1, -1):
            if i % 8 == 7:
                print(f'C {int(-(i / 8) + 9)} * ', end='')

            if black.get_cell_state(i):
                logger.log_comment('B ', False)
            elif white.get_cell_state(i):
                logger.log_comment('W ', False)
            else:
                logger.log_comment('- ', False)

            if i % 8 == 0:
                print('')

    def apply_move(self, board: BitBoard, move):
        board.apply_isolated_move(move)

        # Update board internally
        self.set_for_color(board)
        self.line_cap(board, move)

    def is_game_complete(self):
        player_moves = self.generate_move_mask(self.player_board.bits, self.opponent_board.bits)
        opponent_moves = self.generate_move_mask(self.opponent_board.bits, self.player_board.bits)

        return player_moves == 0 and opponent_moves == 0

    def count_pieces(self) -> int:
        """Returns the count of the total occupied cells on the board"""
        return self.player_board.get_bit_count() + self.opponent_board.get_bit_count()

    def set_for_color(self, board: BitBoard) -> None:
        if board.color == self.player_color:
            self.player_board = board
        else:
            self.opponent_board = board

    def get_for_color(self, p_color: int) -> BitBoard:
        return self.player_board if p_color == self.player_color else self.opponent_board

    # noinspection DuplicatedCode
    def generate_move_mask(self, player_bits: np.uint64, opponent_bits: np.uint64):
        player_bits = np.uint64(player_bits)
        opponent_bits = np.uint64(opponent_bits)

        empty_mask = ~player_bits & ~opponent_bits
        move_mask = np.uint64(0)

        for i in range(self.DIRECTION_COUNT):
            # Finds opponent disks that are adjacent to player disks in current direction
            hold_mask = player_bits

            if self.DIR_INCREMENTS[i] > 0:
                hold_mask = (hold_mask << np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]
            else:
                hold_mask = (hold_mask >> -np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]

            hold_mask = hold_mask & opponent_bits

            for j in range(6):
                if not (j < 6) & (hold_mask != 0):
                    break

                if self.DIR_INCREMENTS[i] > 0:
                    hold_mask = (hold_mask << np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]
                else:
                    hold_mask = (hold_mask >> -np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]

                dir_move_mask = hold_mask & empty_mask
                move_mask |= dir_move_mask
                hold_mask &= (~dir_move_mask & opponent_bits)

        return move_mask

    def generate_moves_priority_queue(self, p_state: BitBoard, o_state: BitBoard):
        queue = PriorityQueue()

        state = self.generate_move_mask(p_state.bits, o_state.bits)
        for i in range(64):
            mask = np.uint64(1 << i)
            if (mask & state) != 0:
                weight = self.WEIGHT_MAP[i]
                move = m.Move(i, weight, False)
                priority_item = m.PrioritizedItem(move.value, move)
                queue.put(priority_item)

        return queue

    # noinspection DuplicatedCode
    def line_cap(self, board: BitBoard, move):
        # Move is assumed to be applied to the board already when this function is applied.
        opp = self.get_for_color(-board.color)

        self_bits = board.bits
        opp_bits = opp.bits

        mask = np.uint64(1 << move.pos)
        f_fin = np.uint64(0)

        for i in range(self.DIRECTION_COUNT):
            to_change = np.uint64(0)

            if self.DIR_INCREMENTS[i] > 0:
                search = (mask << np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]
            else:
                search = (mask >> -np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]

            possibility = opp_bits & search

            while possibility != 0:
                to_change |= possibility
                if self.DIR_INCREMENTS[i] > 0:
                    search = (search << np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]
                else:
                    search = (search >> -np.uint64(self.DIR_INCREMENTS[i])) & self.DIR_MASKS[i]

                if (self_bits & search) != 0:
                    f_fin |= to_change
                    break

                possibility = opp_bits & search

        self_bits |= f_fin
        opp_bits = (~f_fin) & opp_bits

        board.bits = self_bits
        opp.bits = opp_bits

        self.set_for_color(board)
        self.set_for_color(opp)

    @staticmethod
    def __get_sum_weight(p_amt, o_amt):
        """Helper function for evaluate(). Computes a weighted sum for two values.
        Returns 0 if the sum is zero. Otherwise returns 100.0 * (p_amt - o_amt) / (p_amt + o_amt)"""
        if p_amt + o_amt == 0:
            return 0
        sum = p_amt + o_amt
        diff = p_amt - o_amt

        w = 100.0 * diff / sum
        if sum < 0:
            return -abs(w)

        return w

    def evaluate(self):
        p_board = self.player_board
        o_board = self.opponent_board

        p_count = p_board.get_bit_count()
        o_count = o_board.get_bit_count()

        p_moves_possible = self.generate_move_mask(p_board.bits, o_board.bits)
        o_moves_possible = self.generate_move_mask(o_board.bits, p_board.bits)

        p_pos_weight = o_pos_weight = p_corners = o_corners = p_adj_corners = o_adj_corners = 0

        p_corner_mask = np.uint64(p_board.bits & self.CORNER_MASK)
        o_corner_mask = np.uint64(o_board.bits & self.CORNER_MASK)

        p_adj_corner_mask = np.uint64(p_board.bits & self.CORNER_ADJACENT_MASK)
        o_adj_corner_mask = np.uint64(o_board.bits & self.CORNER_ADJACENT_MASK)

        # Count number of adjacent corners for each player & their position weight.
        for i in range(64):
            mask = np.uint64(1 << i)
            weight = self.WEIGHT_MAP[i]

            # Pos weights
            if (p_moves_possible & mask) != 0:
                p_pos_weight += weight
            elif (o_moves_possible & mask) != 0:
                o_pos_weight += weight

            # Corners
            if (p_corner_mask & mask) != 0:
                p_corners += 1
            elif (o_corner_mask & mask) != 0:
                o_corners += 1

            # Adjacent corners
            if (p_adj_corner_mask & mask) != 0:
                p_adj_corners += 1
            elif (o_adj_corner_mask & mask) != 0:
                o_adj_corners += 1

        # Compute weights
        w_stability = self.__get_sum_weight(p_pos_weight, o_pos_weight)
        w_parity = self.__get_sum_weight(p_count, o_count)
        w_corners = self.__get_sum_weight(p_corners, o_corners)
        w_adj_corners = self.__get_sum_weight(p_adj_corners, o_adj_corners)
        w_mobility = self.__get_sum_weight(utils.count_bits(p_moves_possible), utils.count_bits(o_moves_possible))

        # Factors. How important each value is relative to the others.
        # todo: perhaps include additional weight for if a player is forced to pass
        f_corners = 90
        f_adj_corners = 10
        f_mobility = 15
        f_parity = 5
        f_stability = 45

        score = int((f_corners * w_corners) + (f_adj_corners * w_adj_corners) +
                    (f_mobility * w_mobility) + (f_parity * w_parity) + (f_stability * w_stability))

        # End game
        if p_count + o_count == 64:
            if w_parity < 0:
                return -99999999
            else:
                return 99999999

        return score

    def alpha_beta(self, board: 'GameBoard', player: int, depth: int, max_depth: int, alpha, beta, maximizer: bool):
        if depth >= max_depth or board.is_game_complete():
            res = SearchResult(depth, player, board.evaluate())
            return res

        maximizer_board = board.get_for_color(player)
        minimizer_board = board.get_for_color(-player)

        if maximizer:
            queue = board.generate_moves_priority_queue(maximizer_board, minimizer_board)
            max_eval = m.MIN_VAL

            while not queue.empty():
                board.apply_move(maximizer_board, queue.get().move)
                search_result = self.alpha_beta(board, -player, depth + 1, max_depth, alpha, beta, False)
                max_eval = max(max_eval, search_result.score)
                alpha = max(alpha, search_result.score)
                if beta <= alpha:
                    break

            return SearchResult(depth, player, max_eval)
        else:
            queue = board.generate_moves_priority_queue(maximizer_board, minimizer_board)
            min_eval = m.MAX_VAL

            while not queue.empty():
                board.apply_move(maximizer_board, queue.get().move)
                search_result = self.alpha_beta(board, -player, depth + 1, max_depth, alpha, beta, True)
                min_eval = min(min_eval, search_result.score)
                beta = min(beta, search_result.score)
                if beta <= alpha:
                    break

            return SearchResult(depth, player, min_eval)

    def select_move(self, p_color: int, random: bool):
        p = self.get_for_color(p_color)
        o = self.get_for_color(-p_color)

        # test
        max_depth = 4
        p_moves = self.generate_moves_priority_queue(p, o)

        if random:
            return p_moves.get().move

        # Todo: Time
        best_move = m.Move(is_pass=False)
        while not p_moves.empty():
            next_priority = p_moves.get()
            next_move = m.Move(next_priority.move.pos, next_priority.move.value, next_priority.move.isPass)
            new_board = copy.deepcopy(self)

            # Todo: Iterative deepening
            evaluation = self.alpha_beta(new_board, p_color, 1, max_depth, m.MIN_VAL, m.MAX_VAL, True)
            if evaluation.score > best_move.value:
                replacement_move = m.Move(next_move.pos, evaluation.score, False)
                replacement_move.search_result = evaluation
                best_move = replacement_move

            logger.log_comment(f'Evaluated {next_move}')

        logger.log_comment(f'Identified {best_move} as best move.')
        return best_move
