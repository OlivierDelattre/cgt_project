import egttools as egt
import numpy as np
import math
import matplotlib.pyplot as plt

from run import *
from simulate import *


def computeAvgReward():
    avg_def_reward_alphas = []
    avg_coop_reward_alphas = []
    avg_exc_reward_alphas = []
    for alpha in range[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            # in [0.1,0.2] :
        game = CRDWithExecutor(
            strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
            initial_endowment=b,
            population_size=Z,
            group_size=N,
            cost=c,
            risk=r,
            alpha=alpha,
            cooperation_threshold=M,
            enhancement_factor=3,
            pi_t=pi_t,
            pi_e=pi_e,
            n_e=n_e,
            mu=mu)
        game.calculate_payoffs()
        # sd = estimate_stationary_distribution(
        #     game=game,
        #     nb_runs=nb_runs,
        #     transitory=transitory,
        #     nb_generations=nb_generations,
        #     beta=beta,
        #     mu=mu,
        #     Z=Z,
        # )
        # avg_def_reward=
        avg_def_reward_alphas.append(np.mean(game.ce_rewards_d_fines[0]))
        avg_exc_reward_alphas.append(np.mean(game.ce_rewards_d_fines[1]))
        avg_coop_reward_alphas.append(np.mean(game.ce_rewards_d_fines[2]))
        # print(f"{avg_def_reward}, {avg_exc_reward}, {avg_coop_reward} for alpha : {alpha}")  # Avg reward/fine
    return avg_def_reward_alphas, avg_exc_reward_alphas, avg_coop_reward_alphas


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

    # game = CRDWithExecutor(
    #     strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
    #     initial_endowment=b,
    #     population_size=Z,
    #     group_size=N,
    #     cost=c,
    #     risk=r,
    #     alpha=alpha,
    #     cooperation_threshold=M,
    #     enhancement_factor=1,
    #     pi_t=pi_t,
    #     pi_e=pi_e,
    #     n_e=n_e,
    #     mu=mu)

    # payoffs = game.calculate_payoffs()
    #
    # evolver = egt.analytical.StochDynamics(
    #     3,
    #     np.array(payoffs),
    #     pop_size=game.Z,
    #     group_size=game.N,
    #     mu=game.mu)

    # for i in range(egt.calculate_nb_states(4, 3)):
    #     print(egt.sample_simplex(i, 4, 3), " -> ", payoffs.transpose()[i])

    # transition_matrix = evolver.calculate_full_transition_matrix(beta=beta)

    # incredibly slow + bugged in egttools 1.11
    # stationary_distribution = evolver.calculate_stationary_distribution(beta=beta)
    # sd = estimate_stationary_distribution(
    #     game=game,
    #     nb_runs=nb_runs,
    #     transitory=transitory,
    #     nb_generations=nb_generations,
    #     beta=beta,
    #     mu=mu,
    #     Z=Z,
    # )

    strategy_labels = ["Defector", "Executor", "Cooperator"]
    colors = sns.color_palette("viridis", 3)
    fix, ax = plt.subplots(figsize=(8, 5))
    # avg_rewards_fine = computeAvgReward()
    # print(avg_rewards_fine)
    avg_def_reward_alphas, avg_exc_reward_alphas, avg_coop_reward_alphas= computeAvgReward()
    print(avg_coop_reward_alphas)
    # nIgsPerRisk= np.empty([3, 10])
    # np.vstack((nIgsPerRisk,nIs_risk0))
    # [nIgsPerRisk, [nIs_risk0, nIs_risk02, nIs_risk05]]
    # print(nIgsPerRisk)

    # for i, color in enumerate(colors):
    #     ax.plot(np.linspace(0.0, 1.0, num=10), avg_rewards_fine[i], color=color, lw=2, '*--', label='r = 0')
    #     ax.scatter(betas, 1 - coop_level_analytical, marker='x', label="analytical")

    # ax.scatter(betas, 1 - coop_level_analytical, marker='x', label="analytical")
    # ax.scatter(betas, coop_level, marker='o', label="numerical")

    # plt.figure()
    plt.plot(np.arange(0, 1.01, 0.1), avg_def_reward_alphas, label='fine of defectors')
    plt.plot(np.arange(0, 1.01, 0.1), avg_exc_reward_alphas, label='Reward of excecutor')
    plt.plot(np.arange(0, 1.01, 0.1), avg_coop_reward_alphas, label='Reward of cooperator')
    ax.set_ylabel('Average fine / reward', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mixed coefficient (alpha)', fontsize=15, fontweight='bold')
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.tick_params(axis='x', which='both', labelsize=15, width=3)
    # ax.tick_params(axis='y', which='both',
    #                direction='in', labelsize=15, width=3)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label1.set_fontweight('bold')
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_fontweight('bold')
    plt.show()

