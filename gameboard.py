import copy
import collections
import math
import numpy as np
import time
import resource

import color
import logger

from config import Config
from move import Move, PrioritizedMove, PrioritizedMoveQueue
from dataclasses import dataclass


@dataclass
class SearchResult:
    depth: int
    player: int
    score: int


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_value = {0: 1.0}
        self.child_visits = {0: 1.0}


# Inspired by: https://github.com/brilee/python_uct/blob/master/naive_impl.py
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
class UCTNode:
    def __init__(self, game_state, parent=None, move=None, i=0):
        self.game_state: GameBoard = game_state
        self.parent: UCTNode = parent
        self.is_expanded = False
        self.root = self if self.parent is None else self.__find_root()
        self.children = []
        self.i = i  # int, counts move number in gameplay sequence.
        self.move = move
        self.child_value = np.zeros([65], dtype=np.float32)
        self.child_visits = np.zeros([65], dtype=np.float32)

    def __str__(self):
        return f'UCTNode: [Stats=[{self.value} / {self.visits}] | is_expanded={self.is_expanded} | children={len(self.children)}]'

    @property
    def visits(self):
        return self.parent.child_visits[self.i]

    @visits.setter
    def visits(self, value):
        self.parent.child_visits[self.i] = value

    @property
    def value(self):
        return self.parent.child_value[self.i]

    @value.setter
    def value(self, value):
        self.parent.child_value[self.i] = value

    # Probably really inefficient
    def __find_root(self):
        cur = self

        while cur.parent is not None:
            cur = cur.parent

        return cur

    def child_Q(self):  # returns np.array
        return self.child_value / (1 + self.child_visits)

    def child_U(self):  # returns np.array
        n = self.child_visits
        if type(n) is dict:
            n = n[0]

        return math.sqrt(2) * math.sqrt(math.log(n / (1 + self.child_visits)))

    def choose_best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select(self):
        current = self
        while current.is_expanded:
            current = current.choose_best_child()
        return current

    def set_children(self):
        pb = self.game_state.get_for_color(self.game_state.to_play)
        ob = self.game_state.get_for_color(-self.game_state.to_play)
        moves = self.game_state.generate_moves_priority_queue(pb, ob)
        while not moves.empty():
            move = moves.get().move
            if move.isPass:
                continue

            cp = copy.deepcopy(self.game_state)
            cp.apply_move(move)
            self.children.append(UCTNode(cp, parent=self, move=move, i=self.i + 1))

    def expand(self):
        self.set_children()

        if self.children:
            self.is_expanded = True

    def simulate(self):
        cp = copy.deepcopy(self.game_state)

        initial_player_save = cp.to_play

        cur_player = initial_player_save
        while not cp.is_game_complete():
            pb = cp.get_for_color(cur_player)
            ob = cp.get_for_color(-cur_player)
            moves = cp.generate_moves_priority_queue(pb, ob).items
            random_choice = np.random.choice(moves).move
            cp.apply_move(random_choice)
            cur_player = -cur_player

        net_pieces = cp.get_for_color(initial_player_save).get_bit_count() - cp.get_for_color(
            -initial_player_save).get_bit_count()
        return initial_player_save, net_pieces

    def back_propogate(self, initial_player, net_pieces):
        if net_pieces == 0:  # Draw
            player_update_val = 0.5
            opponent_update_val = 0.5
        else:
            if net_pieces > 0:
                player_update_val = 1
                opponent_update_val = 0
            else:
                player_update_val = 0
                opponent_update_val = 1

        # Climb up the tree, to the root.
        cur = self
        while cur.parent is not None:
            # Color of the player who made the move we are evaluating. to_play is looking ahead 1 move.
            cur.visits += 1

            if cur.game_state.to_play == initial_player:
                cur.value += player_update_val
            else:
                cur.value += opponent_update_val

            cur = cur.parent

    def add_child(self, move: Move):
        """Adds child node to known children"""
        if move not in [x.move for x in self.children]:
            self.game_state.apply_move(move)
            self.children.append(UCTNode(self.game_state, parent=self, move=move))


class GameBoard:
    DIRECTION_COUNT = 8
    UNIVERSE = np.uint64(0xffffffffffffffff)

    CORNER_MASK = np.uint64(0x8100000000000081)
    CORNER_ADJACENT_MASK = np.uint64(0x42C300000000C342)

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

    STABILITY_IGNORES = {
        0: [1, 8, 9],
        7: [6, 14, 15],
        56: [57, 48, 49],
        63: [62, 55, 54]
    }

    from bitboard import BitBoard
    def __init__(self, config: Config, player: BitBoard, opponent: BitBoard):
        self.config = config
        self.player_color = player.color
        self.opponent_color = -player.color
        self.player_board = player
        self.opponent_board = opponent
        self.move_history: list[Move] = []
        self.to_play = color.BLACK  # Keeps track of who moves next. Black always moves first in Othello

    def __str__(self):
        return f'--- GameBoard ---\n' \
               f'Player Board: {self.player_board}\n' \
               f'Opponent Board: {self.opponent_board}\n' \
               f'Config: {self.config}\n' \
               f'Available moves primary:\n{self.generate_moves_priority_queue(self.player_board, self.opponent_board)}' \
               f'Available moves opponent:\n{self.generate_moves_priority_queue(self.opponent_board, self.player_board)}'

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

    def print_move_history(self):
        logger.log_comment(f'Move History:')
        for i in range(len(self.move_history)):
            logger.log_comment(f'Move #{i + 1}: {self.move_history[i]}')

    def is_valid(self, move):
        """Determines whether a given move is valid."""
        moves = self.generate_move_mask(self.get_for_color(move.color).bits, self.get_for_color(-move.color).bits)

        # Pass moves
        if moves == 0:
            return move.pos == -1

        return (np.uint64(1 << move.pos) & moves) != 0

    def apply_move(self, move: Move):
        self.move_history.append(move)
        self.to_play = -move.color

        if move.isPass:
            return

        # Update board internally
        board = self.get_for_color(move.color)
        board.apply_isolated_move(move)

        self.update_board(board)
        self.line_cap(board, move)

    def is_game_complete(self):
        player_moves = self.generate_move_mask(self.player_board.bits, self.opponent_board.bits)
        opponent_moves = self.generate_move_mask(self.opponent_board.bits, self.player_board.bits)

        return player_moves == 0 and opponent_moves == 0

    def count_pieces(self) -> int:
        """Returns the count of the total occupied cells on the board"""
        return self.player_board.get_bit_count() + self.opponent_board.get_bit_count()

    def update_board(self, board: BitBoard) -> None:
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
        queue = PrioritizedMoveQueue()

        state = self.generate_move_mask(p_state.bits, o_state.bits)
        for i in range(64):
            mask = np.uint64(1 << i)
            if (mask & state) != 0:
                weight = self.WEIGHT_MAP[i]
                move = Move(p_state.color, i, weight, False)
                priority_item = PrioritizedMove(move.value, move)
                queue.put(priority_item)

        if len(queue.items) == 0:
            default = Move(color=p_state.color)
            queue.put(PrioritizedMove(default.value, default))

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

        self.update_board(board)
        self.update_board(opp)

    def UCTSearch(self, num_reads):
        root = UCTNode(self, parent=DummyNode())
        for i in range(num_reads):
            leaf = root.select()
            leaf.expand()
            i, n = leaf.simulate()
            leaf.back_propogate(i, n)
        return max(root.children, key=lambda x: x.visits)

    def select_move(self):
        reads = 1000
        tick = time.time()
        result = self.UCTSearch(reads)
        tock = time.time()

        logger.log_comment(f'Took {tock - tick:.2f}s to run {reads} times.')
        logger.log_comment(f'Consumed {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000}MB memory.')
        logger.log_comment(f'Result: {result}')

        return result.move

    def select_random_move(self, p_color: int):
        p = self.get_for_color(p_color)
        o = self.get_for_color(-p_color)

        p_moves = self.generate_moves_priority_queue(p, o).items
        if len(p_moves) == 0:
            logger.log_comment(f'No moves to make. Returning pass move.')
            return Move(color=p_color)

        return p_moves[np.random.randint(len(p_moves))].move
