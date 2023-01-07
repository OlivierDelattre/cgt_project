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
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
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
                nIs_risk0[alpha] = group_achievement
            elif risk == 0.2:
                nIs_risk02[alpha] = group_achievement
            else:
                nIs_risk05[alpha] = group_achievement
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
    n_e = 1
    alpha = 1.
    mu = 1 / Z
    beta = 5.
    transitory = 10 ** 2  # num of steps before we start counting
    nb_generations = 10 ** 4  # num of steps where we do count
    nb_runs = 10  # num of different runs we average over

    strategy_labels = ["Defector", "Executor", "Cooperator"]
    colors = sns.color_palette("viridis", 3)
    fix, ax = plt.subplots(figsize=(8, 5))
    nIgsPerRisk = computeNIs()
    # nIgsPerRisk= np.empty([3, 10])
    # np.vstack((nIgsPerRisk,nIs_risk0))
    # [nIgsPerRisk, [nIs_risk0, nIs_risk02, nIs_risk05]]
    # print(nIgsPerRisk)
    for i, color in enumerate(colors):
        ax.plot(np.linspace(0.0, 1.0, num=10), nIgsPerRisk[i], color=color, lw=2)
    ax.set_ylabel('Institution prevalence (nI)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mixed coefficient (alpha)', fontsize=15, fontweight='bold')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='both', labelsize=15, width=3)
    ax.tick_params(axis='y', which='both',
                   direction='in', labelsize=15, width=3)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    plt.show()
