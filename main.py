import sys
import argparse

import output
import time
from config import Config
import color
import numpy as np

import logger
from gameboard import GameBoard
from bitboard import BitBoard


# cfg = config.Config()

def arg_parser():
    parser = argparse.ArgumentParser(description='PyOthello for CSCI312 - Harry Burnett')

    parser.add_argument('--interactive', action='store_true',
                        help='Include to play PyOthello interactively.')
    parser.add_argument('--random', action='store_true',
                        help='Include to have the opponent play random moves')
    parser.add_argument('--time', type=int, help='The total amount of time to allot for the game, in seconds.')
    return parser


def print_args(a):
    logger.log_comment('--- Arguments ---')
    logger.log_comment('Interactive: '.ljust(15, ' ') + str(a.interactive))
    logger.log_comment('Random: '.ljust(15, ' ') + str(a.random))
    logger.log_comment('Time: '.ljust(15, ' ') + str(a.time))


def load_config(a):
    """Loads config from arguments"""
    return Config(a.interactive, a.random, a.time)


# Load and display program arguments
parser = arg_parser()
args = parser.parse_args()
print_args(args)

# Load config from arguments
cfg = load_config(args)

# Init to black / white as default. Change based on results from prompt.
p_color = color.BLACK
o_color = color.WHITE

logger.log_comment(f'Initialized default player color to {p_color}')
logger.log_comment(f'Initialized default opponent color to {o_color}')

while True:
    logger.log_comment('Init agent color [I B] or [I W]')
    init = input()

    if not init == 'I B' and not init == 'I W':
        logger.log_comment(f'Expected: \'I B\' or \'I W\' -- received: {init}')
        continue

    if init == 'I B':
        print('R B')
    else:
        p_color = color.WHITE
        o_color = color.BLACK

        logger.log_comment(f'Changed player color to {p_color}')
        logger.log_comment(f'Changed opponent color to {o_color}')

        print('R W')
    break

# Init default bitboards
p_board = BitBoard(p_color)
o_board = BitBoard(o_color)
g_board = GameBoard(cfg, p_board, o_board)

logger.log_comment('Initialized board.')
logger.log_comment(f'AI configuration:'.ljust(25, ' ') + str(p_board))
logger.log_comment(f'Opponent configuration:'.ljust(25, ' ') + str(o_board))
g_board.draw()

play_as_black = p_color == color.BLACK
black_turn = True

# Primary game loop
move_count = 1
while not g_board.is_game_complete():
    p_turn = play_as_black and black_turn or not play_as_black and not black_turn
    p_board = g_board.player_board
    o_board = g_board.opponent_board

    designator = 'Black Turn' if black_turn else 'White Turn'
    prompt = f'Move #{move_count} -- {designator}'
    logger.log_comment(prompt)

    if p_turn:
        start = time.time()
        move = g_board.select_move(p_color, False)
        end = time.time()

        print(f'Evaluated best move in {end - start}s')

        if not move.isPass:
            g_board.apply_move(p_board, move)

        inpt = output.out_move(p_color, move, True)
    else:
        if cfg.interactive:
            possible_moves = g_board.generate_move_mask(o_board.bits, p_board.bits)
            inpt = input()
            move = output.to_move(inpt)

            # Checks for valid move or valid pass if that was registered.
            valid = possible_moves > 0 and (np.uint64((1 << move.pos)) & possible_moves) != 0 or \
                    (move.isPass and possible_moves == 0)
        else:
            move = g_board.select_move(o_color, cfg.random)

        if not move.isPass:
            g_board.apply_move(o_board, move)

        inpt = output.out_move(o_color, move, True)

    if p_turn:
        print(inpt)

    g_board.draw()
    logger.log_comment(g_board.player_board)
    logger.log_comment(g_board.opponent_board)

    logger.log_comment(f'Score (BLACK): {g_board.get_for_color(color.BLACK).get_bit_count()}')
    logger.log_comment(f'Score (WHITE): {g_board.get_for_color(color.WHITE).get_bit_count()}')

    black_turn = not black_turn
    move_count += 1

# Output result of game
f_black = g_board.get_for_color(color.BLACK).get_bit_count()
logger.log_comment('---GAME FINISHED---')
print(f_black)
