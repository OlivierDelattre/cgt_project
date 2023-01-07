import egttools as egt
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

# from run import CRDWithExecutor, Cooperator, Defector, Executor
from run import *
if __name__ == '__main__': 

    Z = 100         # Population size
    N  = 4           # Group size
    b  = 1.           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.3         # Minimum collective contribution
    M  = 3.           # OR Minimum number of cooperators
    #r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 1
    alpha = 1.
    mu    = 1/Z
    beta = 5.
    #delta = 2

    ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ######
    ##      IN FIG A,B,D: r = 0.2               IN FIG C: r = 0.5   ##
    ##      IN FIG B: delta = 2                 IN FIG D: delta = 3 ##

    #FIG A
    for delta in []: #[2, 3, 4]:
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            r = 0.2
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=r,
                alpha=alpha,
                cooperation_threshold=M,
                enhancement_factor=delta,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu,
                incentive=('local', 'flexible'))

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

            print(f'delta={delta}_alpha={alpha}_mu={mu}_ne={n_e} => ', group_achievement)

            with open(f'fig4/fig4A_delta={delta}_alpha={alpha}_mu={mu}_ne={n_e}.pickle', 'wb') as f:
                pickle.dump([payoffs, stationary_distribution, group_achievement], f)

    #FIG B
    for alpha in [0, 0.2, 1]:
        for r in [0.22, 0.24, 0.26, 0.28]:
            delta = 2
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=r,
                alpha=alpha,
                cooperation_threshold=M,
                enhancement_factor=delta,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu,
                incentive=('local', 'flexible'))

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

            print(f'delta={delta}_alpha={alpha}_mu={mu}_ne={n_e} => ', group_achievement)

            with open(f'fig4/fig4B_r={r}_alpha={alpha}_mu={mu}_ne={n_e}.pickle', 'wb') as f:
                pickle.dump([payoffs, stationary_distribution, group_achievement], f)

    #FIG C
    for alpha in []:#[0, 0.2, 1]:
        for delta in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]:
            r = 0.5
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=r,
                alpha=alpha,
                cooperation_threshold=M,
                enhancement_factor=delta,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu,
                incentive=('local', 'flexible'))

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

            print(f'delta={delta}_alpha={alpha}_mu={mu}_ne={n_e} => ', group_achievement)

            with open(f'fig4/fig4C_delta={delta}_alpha={alpha}_mu={mu}_ne={n_e}.pickle', 'wb') as f:
                pickle.dump([payoffs, stationary_distribution, group_achievement], f)

    # Fig 4
    for alpha in [0, 0.2, 1]:
        for mu in [0.02, 0.04, 0.06, 0.08]: #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            r = 0.2
            delta = 3
            game = CRDWithExecutor(
                strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                initial_endowment=b,
                population_size=Z,
                group_size=N,
                cost=c,
                risk=r,
                alpha=alpha,
                cooperation_threshold=M,
                enhancement_factor=delta,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=n_e,
                mu=mu,
                incentive=('local', 'flexible'))

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

            print(f'delta={delta}_alpha={alpha}_mu={mu}_ne={n_e} => ', group_achievement)

            with open(f'fig4/fig4D_delta={delta}_alpha={alpha}_mu={mu}_ne={n_e}.pickle', 'wb') as f:
                pickle.dump([payoffs, stationary_distribution, group_achievement], f)