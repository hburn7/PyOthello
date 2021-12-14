import color
import logger
import utils
import move
import color as c


def write_init(term_input: str):
    """Writes the appropriate input to the console. Converts 'I B' to 'R B', etc."""
    if term_input == 'I B':
        print('R B')
    elif term_input == 'I W':
        print('R W')
    else:
        print(term_input)

def out_move(p_color: int, move: move.Move, log_comment: bool):
    color_char = 'B' if p_color == color.BLACK else 'W'

    if move.pos < 0 or move.pos > 63:
        return color_char

    # row_col[0] is the row, [1] is the column
    row_col = utils.pos_to_row_col(move.pos)
    ret = f'{color_char} {row_col[1]} {row_col[0]}'

    if log_comment:
        logger.log_comment(f'Converted pos {move.pos} to {ret}')

    return ret

def to_move(s: str, log_comment: bool) -> move.Move:
    """Returns a move for a given (valid) input. Returns a pass if the input is invalid"""

    # Assumes input is in format 'W a 1' or 'B h 8', etc.
    x = 7 - utils.col_from_char(s[2])
    y = 8 - int(s[4])

    pos = (y * 8) + x

    if pos < 0 or pos > 63:
        logger.log_comment(f'Received input \'{s}\' (deemed invalid) -> returns pass move.')
        return move.Move()

    if log_comment:
        logger.log_comment(f'Converted {s} to pos: {pos}')
    return move.Move(pos, is_pass=False)
