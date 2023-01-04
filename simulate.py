import numpy as np
import math
import matplotlib.pyplot as plt

from run import CRDWithExecutor, Cooperator, Defector, Executor

class simulator():
    def __init__(self, game):
        self.game = game

    def 


if __name__ == '__main__': 

    Z  = 10         # Population size
    N  = 4           # Group size
    b  = 1           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.5         # Minimum collective contribution
    M  = 3           # OR Minimum number of cooperators
    r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 0.25
    alpha = 1
    mu    = 1/Z

    game = CRDWithExecutor(strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                    initial_endowment=b,
                    population_size=Z,
                    group_size=N,
                    cost=c,
                    risk=r,
                    alpha=alpha,
                    cooperation_threshold=M,
                    enhancement_factor=1,
                    pi_t=pi_t,
                    pi_e=pi_e,
                    n_e=n_e,
                    mu=mu)