import egttools as egt
import numpy as np
import math
import matplotlib.pyplot as plt

from run import *
from simulate import *


def computeNIs():
    nIs_risk0 = np.zeros(11)
    nIs_risk02 = np.zeros(11)
    nIs_risk05 = np.zeros(11)
    for i,alpha in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        for risk in [0.0, 0.2, 0.5]:
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=risk,
                alpha=alpha,
                cooperation_threshold=M,
                enhancement_factor=1,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu)
            sd = estimate_stationary_distribution(
                game=game,
                nb_runs=nb_runs,
                transitory=transitory,
                nb_generations=nb_generations,
                beta=beta,
                mu=mu,
                Z=Z,
            )
            group_achievement = sum([
                sd[i] * game.aI(i) for i in range(len(sd))
            ])
            if risk == 0.0:
                nIs_risk0[i] = group_achievement
            elif risk == 0.2:
                nIs_risk02[i] = group_achievement
            else:
                nIs_risk05[i] = group_achievement
            print(f"{group_achievement} for alpha : {alpha}")  # institution prevalence
    return nIs_risk0, nIs_risk02, nIs_risk05


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
    n_e = 2
    alpha = 1.
    mu = 1 / Z
    beta = 5.
    transitory = 10 ** 2  # num of steps before we start counting
    nb_generations = 10 ** 4  # num of steps where we do count
    nb_runs = 10  # num of different runs we average over

    strategy_labels = ["Defector", "Executor", "Cooperator"]
    fix, ax = plt.subplots(figsize=(8, 5))
    nIs_delta2, nIs_delta3, nIs_delta4 = computeNIs()
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta2, '*--', label='risk = 0.5')
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta3, '*--', label='\u03B4 = 0.2')
    plt.plot(np.arange(0, 1.01, 0.1), nIs_delta4, '*--', label='\u03B4 = 0.0')
    ax.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    ax.set_ylabel('Institution prevalence, \u03B7I', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mixed coefficient, \u03B1', fontsize=15, fontweight='bold')
    plt.show()
