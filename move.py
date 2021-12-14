import numpy as np
from dataclasses import dataclass, field
from queue import PriorityQueue

from typing import List, Optional

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
class PrioritizedMove:
    priority: int
    move: Move = field(compare=False)

class PrioritizedMoveQueue(PriorityQueue):
    def __init__(self, items: List[PrioritizedMove]):
        super().__init__()
        for x in items:
            self.put(x)
        self.items = items

    def __init__(self):
        super().__init__()
        self.items = []

    def __str__(self):
        s = ''
        for item in self.items:
            s += f'PrioritizedMove: Priority: {item.priority} | {item.move}\n'
        return s

    def put(self, item: PrioritizedMove, block: bool = ..., timeout: Optional[float] = ...) -> None:
        super().put(item, block, timeout)
        self.items.append(item)

