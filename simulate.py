import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
import egttools as egt
import pickle

from run import CRDWithExecutor, Cooperator, Defector, Executor

verbosity = 0

def step(game, evolver, current_population, beta, mu, Z, payoffs):
    #Sample 2 random individuals
    #population = current_population.copy()
    population = current_population

    sample = sp.stats.multivariate_hypergeom.rvs(m=current_population, n=2)
    idvs = [0 for _ in range(sample[0])]+[1 for _ in range(sample[1])]+[2 for _ in range(sample[2])]
    np.random.shuffle(idvs)
    idv0, idv1 = idvs

    if np.random.rand() < mu:
        # Random mutation (change strategy)
        population[idv0] -= 1
        population[np.random.randint(0, 3)] += 1
    elif idv0 == idv1:
        return population
    else:
        #calculate fitness difference
        fitness_diff = evolver.full_fitness_difference_group(idv0, idv1, population_state=current_population)
        p = (1 + np.e**(beta*fitness_diff))**-1
        if np.random.rand() < p:
            # Adopt strategy of opponent
            population[idv0] -= 1
            population[idv1] += 1
    
    return population

def estimate_stationary_distribution(game:CRDWithExecutor, nb_runs, transitory, nb_generations, beta, mu, Z):
    # Keep a sum of stationary distribution of each run (which we divide be #runs)
    num_population_states = egt.calculate_nb_states(game.Z, 3)

    stationary_distribution_sum = np.zeros((num_population_states,), dtype=np.float64)
    for run in range(nb_runs):
        print(f"\t\t--- Run {run} ---")
        current_population = egt.sample_simplex(
            np.random.randint(0, num_population_states),
            game.Z,
            3)
        game.set_population_state(current_population)
        payoffs = game.calculate_payoffs()
        evolver = egt.analytical.StochDynamics(
            3, 
            np.array(payoffs), 
            pop_size=game.Z, 
            group_size=game.N, 
            mu=game.mu)

        run_count = np.zeros((num_population_states,), dtype=np.float64)
        print(f"Initial state:\t {current_population}")

        for _ in range(transitory):
            current_population = step(game, evolver, current_population, beta, mu, Z, payoffs)
            game.set_population_state(current_population)
        print(f"After transitory period:\t {current_population}")

        for s in range(nb_generations-transitory):
            if s % 100 == 0:
                print("Step ", s, end='\r')
            current_population = step(game, evolver, current_population, beta, mu, Z, payoffs)
            state = egt.calculate_state(game.Z, current_population)
            run_count[state] += 1
            game.set_population_state(current_population)
        print(f"Final state:\t {current_population}")

        stationary_distribution_sum += run_count/(nb_generations-transitory)

    stationary_distribution = stationary_distribution_sum / nb_runs
    return stationary_distribution

if __name__ == '__main__': 

    Z  = 100         # Population size
    N  = 4           # Group size
    b  = 1           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.5         # Minimum collective contribution
    M  = 3           # OR Minimum number of cooperators
    r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 0.25
    alpha = 0
    mu    = 1/Z

    beta = 5   # Selection strength
    transitory = 10**2      # num of steps before we start counting
    nb_generations = 10**4  # num of steps where we do count
    nb_runs = 10            # num of different runs we average over

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

    sd = estimate_stationary_distribution(
        game=game,
        nb_runs=nb_runs,
        transitory=transitory,
        nb_generations=nb_generations,
        beta=beta,
        mu=mu,
        Z=Z,
    )

    with open(f"alpha={alpha}.pickle", 'wb') as f:
        pickle.dump(sd, f)