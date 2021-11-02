from dataclasses import dataclass

@dataclass
class Config:
    '''Config for the application. Responsible for passing data such as program args
    to the rest of the application.'''
    interactive: bool
    random: bool
    game_time: int