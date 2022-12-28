import egttools as egt
import numpy as np

Z = 100         # Population size
N = 10          # Group size
b = 10          # Endowment (individual's money/funds/...)
c = 1           # Amount of money individuals contribute
Mc = 5          # Minimum collective contribution
M = 5           # OR Minimum number of cooperators
r = 0.5         # If minimum is not met: All group participants lose their endowment
                #   with probability r
                # If minimum is met: individuals retain their endowments

class CRDGameExtended(egt.games.CRDGame):
    pass

class Cooperator(egt.behaviors.CRD.AbstractCRDStrategy):
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return c

    def type(self):
        return "Cooperator"

class Defector(egt.behaviors.CRD.AbstractCRDStrategy):
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return 0

    def type(self):
        return "Defector"

class Executor(egt.behaviors.CRD.AbstractCRDStrategy):
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return c + 'some extra'

    def type(self):
        return "Executor"

strategies = [
    egt.behaviors.NormalForm.CRD.TwoActions.Defector,
    egt.behaviors.NormalForm.TwoActions.Cooperator,
]