import egttools as egt
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

from run import CRDWithExecutor, Cooperator, Defector, Executor

if __name__ == '__main__': 

    Z = 100         # Population size
    N  = 4           # Group size
    b  = 1.           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.3         # Minimum collective contribution
    M  = 3.           # OR Minimum number of cooperators
    r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 0.25
    alpha = 1.
    mu    = 1/Z
    beta = 5.

    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        game = CRDWithExecutor(
            strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
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
        beta = 5
        payoffs = game.calculate_payoffs()

        evolver = egt.analytical.StochDynamics(
            3, 
            np.array(payoffs), 
            pop_size=game.Z, 
            group_size=game.N, 
            mu=game.mu)
     
    #incredibly slow + bugged in egttools 1.11
        stationary_distribution = evolver.calculate_stationary_distribution(beta=beta)
        group_achievement = sum([
            stationary_distribution[i]*game.aG(i) for i in range(len(stationary_distribution))
        ])

        print(f'r={r}_alpha={alpha} => ', group_achievement)

        with open(f'fig2A_r={r}_alpha={alpha}.pickle', 'wb') as f:
            pickle.dump([payoffs, stationary_distribution, group_achievement], f)

    #strategy_labels = ["Defector", "Executor", "Cooperator"]
    #fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    #G = egt.plotting.draw_stationary_distribution(strategy_labels,
    #                                            1/Z, fixation_probabilities, stationary_distribution,
    #                                            node_size=600, 
    #                                            font_size_node_labels=8,
    #                                            font_size_edge_labels=8,
    #                                            font_size_sd_labels=8,
    #                                            edge_width=1,
    #                                            min_strategy_frequency=-0.01, 
    #                                            ax=ax)
    plt.axis('off')
    plt.show() # display