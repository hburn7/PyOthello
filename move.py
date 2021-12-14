import numpy as np
from dataclasses import dataclass, field
import gameboard

MIN_VAL = np.iinfo('int32').min
MAX_VAL = np.iinfo('int32').max


class Move:
    def __init__(self, pos: int = -1, value: int = MIN_VAL, is_pass: bool=True,
                 search_res=None):
        self.pos = pos
        self.value = value
        self.isPass = is_pass
        self.search_result = search_res

    def __str__(self):
        return f'Move [pos: {self.pos} | value: {self.value} | pass: {self.isPass} | search_result: {self.search_result}]'


# Loose wrapper for Move, allows use in a priority queue
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    move: Move = field(compare=False)
