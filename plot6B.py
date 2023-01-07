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
        avg_def_reward=np.mean(game.ce_rewards_d_fines[0])
        avg_exc_reward = np.mean(game.ce_rewards_d_fines[1])
        avg_coop_reward = np.mean(game.ce_rewards_d_fines[2])
        avg_def_reward_alphas.append(avg_def_reward)
        avg_exc_reward_alphas.append(avg_exc_reward)
        avg_coop_reward_alphas.append(avg_coop_reward)
        print(f"def:{avg_def_reward}, exc:{avg_exc_reward}, coop:{avg_coop_reward} for alpha : {alpha}")
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
    strategy_labels = ["Defector", "Executor", "Cooperator"]

    fix, ax = plt.subplots(figsize=(8, 5))
    # avg_rewards_fine = computeAvgReward()
    # print(avg_rewards_fine)
    avg_def_reward_alphas, avg_exc_reward_alphas, avg_coop_reward_alphas= computeAvgReward()
    print(avg_coop_reward_alphas)
    plt.plot(np.arange(0, 1.01, 0.1), avg_def_reward_alphas,'*--', label='Fine of defectors')
    plt.plot(np.arange(0, 1.01, 0.1), avg_exc_reward_alphas,'*--', label='Reward of excecutor')
    plt.plot(np.arange(0, 1.01, 0.1), avg_coop_reward_alphas,'*--',label='Reward of cooperator')
    ax.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    ax.set_ylabel('Average fine / reward', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mixed coefficient (alpha)', fontsize=15, fontweight='bold')
    plt.show()

