import egttools as egt
import numpy as np
import math
import matplotlib.pyplot as plt

from run import *
from simulate import *


def computeNIs():
    nIs_delta2 = np.zeros(10)
    nIs_delta3 = np.zeros(10)
    nIs_delta4 = np.zeros(10)
    for alpha in range(0, 10):
        for delta in [2, 3, 4]:
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha / 10), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=r,
                alpha=alpha / 10,
                cooperation_threshold=M,
                enhancement_factor=delta,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu)
            # sd = estimate_stationary_distribution(
            #     game=game,
            #     nb_runs=nb_runs,
            #     transitory=transitory,
            #     nb_generations=nb_generations,
            #     beta=beta,
            #     mu=mu,
            #     Z=Z,
            # )
            payoffs = game.calculate_payoffs()
            evolver = egt.analytical.StochDynamics(
                game.nb_strategies_,
                np.array(payoffs),
                pop_size=game.Z,
                group_size=game.N,
                mu=game.mu)

            sd = evolver.calculate_stationary_distribution(beta=beta)
            group_achievement = sum([
                sd[i] * game.aI(i) for i in range(len(sd))
            ])
            if delta == 2:
                nIs_delta2[alpha] = group_achievement
            elif delta == 3:
                nIs_delta3[alpha] = group_achievement
            else:
                nIs_delta4[alpha] = group_achievement
            print(f"{group_achievement} for alpha : {alpha / 10}")  # institution prevalence
    return nIs_delta2, nIs_delta3, nIs_delta4


if __name__ == '__main__':
    Z = 100  # Population size
    N = 4  # Group size
    b = 1.  # Endowment (individual's money/funds/...)
    c = 0.1  # Amount of money individuals contribute
    Mc = 0.3  # Minimum collective contribution
    M = 3.  # OR Minimum number of cooperators
    r = 0.2  # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 1
    # alpha = 1.
    mu = 1 / Z
    beta = 5.
    transitory = 10 ** 2  # num of steps before we start counting
    nb_generations = 10 ** 4  # num of steps where we do count
    nb_runs = 10  # num of different runs we average over
    strategy_labels = ["Defector", "Executor", "Cooperator"]
    # ---Plot
    fix, ax = plt.subplots(figsize=(8, 5))
    nIs_delta2, nIs_delta3, nIs_delta4 = computeNIs()
    # test = np.empty(shape=11)
    # test2 = np.empty(shape=11)
    # test3 = np.empty(shape=11)
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta2, '*--', label='\u03B4 = 2')
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta3, '*--', label='\u03B4 = 3')
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta4, '*--', label='\u03B4 = 4')
    ax.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    ax.set_ylabel('Institution prevalence, \u03B7I', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mixed coefficient, \u03B1', fontsize=15, fontweight='bold')
    plt.show()
