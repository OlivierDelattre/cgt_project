import egttools as egt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#import seaborn as sns
from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex


def theta(x): return 0 if x < 0 else 1


class CRDWithExecutor():

	#### -----------------------------------------------------------------------------------------------------------
	#### ---- Adapted from https://github.com/Socrats/EGTTools/blob/stable/src/egttools/games/informal_risk.py -----
	#### -----------------------------------------------------------------------------------------------------------
    
    def __init__(self, strategies: list, initial_endowment, population_size: int, group_size: int, cost: float, risk:float, 
                alpha: float, cooperation_threshold: int, enhancement_factor:float, pi_t: float, pi_e: float, n_e: int, mu: int, incentive:str="fixed"):

        self.N = group_size                     # Size of groups
        self.Z = population_size                # Size of population
        self.M = cooperation_threshold          # Number of cooperators needed for success
        
        self.c = cost                           # Cost of cooperating
        self.r = risk                           # Risk of losing everything if num cooperators is not reached
        self.b = initial_endowment              # Initial funds of each individual

        self.alpha = alpha                      # Positive/Negative incentive trade-off
        self.delta = enhancement_factor         # Enhancement_factor?

        self.pi_t = pi_t                        # Cost of contributing to sanction pool
        self.pi_e = pi_e                        # Support from Executors?
        self.n_e  = n_e
        self.mu   = mu
        self.beta = 5
       
        self.strategies_    = strategies        # List of strategies
        self.nb_strategies_ = len(strategies)   # Number of strategies

        self.ic = np.random.randint(self.Z)
        self.ie = np.random.randint(self.Z-self.ic)
        self.id = Z - self.ie - self.ic

        self.incentive  = incentive
        self.payoffs_   = np.zeros(shape=(self.nb_strategies_, self.nb_strategies_), dtype=np.float64)


    #def transition(l, r):

        #return (1 - self.mu) * (l/self.Z) * (r/(self.Z-1)) * (1 + np.exp(-self.beta) )**-1 + self.mu * (l/(2*self.Z))

    #def transition_cooperator():

        #return transition(self.ic, self.ie) + transition()








    
    
    def Delta(self, je):

        delta = 0

        if je > self.N * 0.25:
            delta = theta(je - self.n_e)
        elif self.ie > self.Z * 0.25:
            delta = theta(self.ie - self.n_e)

        return delta


    # base 

    def base_defector_payoff(self, jc):
        return self.b * theta(jc - self.M) + (1 - self.r)*self.b*(1 - theta(jc - self.M))

    def base_cooperator_payoff(self, jc):
        return self.base_defector_payoff(jc) - self.c


    # fixed incentive 

    def defector_fixed_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc+je) - (1 - self.alpha)*pi_e*self.Delta(je)

    def cooperator_fixed_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc+je) + self.alpha*self.pi_e*self.Delta(je) - self.c

    def executor_fixed_incentives_payoff(self, jc, je):
    	return self.cooperator_fixed_incentives_payoff(jc, je) - self.pi_t

    
    # flexible incentive 

    def defector_flexible_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) - (1 - self.alpha)*((self.pi_t * je * self.delta)/(self.N - jc - je))*self.delta(je)

    def cooperator_flexible_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) + alpha *((self.pi_t * je * self.delta)/(jc + je))*self.delta(je) - self.c

    def executor_flexible_incentives_payoff(self, jc, je):
    	return self.cooperator_flexible_incentives_payoff(jc, je) - self.pi_t


    # average payoffs fixed

    def defector_average_payoffs_fixed(self):

    	fd = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fd += ((math.comb(self.ic, jc) * math.comb(self.ie, je) * math.comb(self.Z-self.ic-self.ie-1, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.defector_fixed_incentives_payoff(jc, je)

    	return fd 


    def cooperator_average_payoffs_fixed(self):

    	fc = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fc += ((math.comb(self.ic-1, jc) * math.comb(self.ie, je) * math.comb(self.Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.cooperator_fixed_incentives_payoff(jc+1, je)

    	return fc


    def executor_average_payoffs_fixed(self):

    	fe = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fe += ((math.comb(self.ic, jc) * math.comb(self.ie-1, je) * math.comb(self.Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.executor_fixed_incentives_payoff(jc, je+1)

    	return fe



    # average payoffs flexible

    def defector_average_payoffs_flexible(self):

    	fd = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fd += ((math.comb(self.ic, jc) * math.comb(self.ie, je) * math.comb(self.Z-self.ic-self.ie-1, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.defector_flexible_incentives_payoff(jc, je)

    	return fd 


    def cooperator_average_payoffs_flexible(self):

    	fc = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fc += ((math.comb(self.ic-1, jc) * math.comb(self.ie, je) * math.comb(self.Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.cooperator_flexible_incentives_payoff(jc+1, je)

    	return fc


    def executor_average_payoffs_flexible(self):

    	fe = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fe += ((math.comb(self.ic, jc) * math.comb(self.ie-1, je) * math.comb(self.Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(self.Z-1, self.N-1)) * self.executor_flexible_incentives_payoff(jc, je+1)

    	return fe


    def calculate_payoffs(self) -> np.ndarray:  # Array of shape (nb_strategies, nb_states_)

        payoffs = np.zeros((self.nb_strategies_, self.nb_strategies_))

        for i in range(self.nb_strategies_):

            if self.incentive == 'fixed':
                PI_D = self.defector_average_payoffs_fixed()
                PI_E = self.executor_average_payoffs_fixed()
                PI_C = self.cooperator_average_payoffs_fixed()
            
            else:
                PI_D = self.defector_average_payoffs_flexible()
                PI_E = self.executor_average_payoffs_flexible()
                PI_C = self.cooperator_average_payoffs_flexible()
            
            payoffs[0, i] = PI_D
            payoffs[1, i] = PI_E
            payoffs[2, i] = PI_C

        self.payoffs_ = payoffs

        return self.payoffs_


    def payoffs(self)->np.ndarray:
        return self.payoffs_

    def save_payoffs(self, file_name:str)->None:
        with open(file_name, 'w') as f:
            f.write('Payoffs for each type of player and each possible state:\n')
            f.write(f'rows: {" ,".join([strategy.type() for strategy in self.strategies_])}\n')
            f.write('cols: all possible group compositions starting at (0, 0, ..., group_size)\n')
            f.write(f'{self.payoffs_}')
            f.write(f'group_size = {self.group_size_}\n')
            f.write(f'cost = {self.c_}\n')
            f.write(f'multiplying_factor = {self.r_}\n')

    def nb_strategies(self)->int:
        return self.nb_strategies_

    def __str__(self)->str:
        return f"CRDWithExecutor_{self.group_size}_{self.cost}"

    def type(self)->str:
        return "CRDWithExecutor"

    
class Cooperator():
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return self.c

    def type(self):
        return "Cooperator"

class Defector():
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return 0

    def type(self):
        return "Defector"

class Executor():
    def __init__(self, c, b, pi_t, pi_e, alpha):
        self.c = c
        self.endowment = b
        self.pi_t = pi_t        # Investment to sanctioning pool
        self.pi_e = pi_e        # positive incentive on cooperators
        self.alpha = alpha      # positive/negative incentive trade-off

    def get_action(self, time_step:int, group_contributions_prev:int):
        return self.c + self.pi_t

    def type(self):
        return "Executor"


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
                    enhancement_factor=0.1,
                    pi_t=pi_t,
                    pi_e=pi_e,
                    n_e=n_e,
                    mu=mu)

    payoffs = game.calculate_payoffs()

    strategy_labels = ["Defector", "Executor", "Cooperator"]

    print(payoffs)

    fig, ax = plt.subplots(figsize=(10,8))

    simplex, gradients, roots, roots_xy, stability = plot_replicator_dynamics_in_simplex(payoffs, ax=ax)

    plot = (simplex.draw_triangle()
               .draw_gradients(density=1)
               .add_colorbar(label='gradient of selection')
               .add_vertex_labels(strategy_labels, epsilon_bottom=0.12)
               .draw_stationary_points(roots_xy, stability)
               .draw_scatter_shadow(lambda u, t: egt.analytical.replicator_equation(u, payoffs), 100, color='gray', marker='.', s=0.1)
              )

    ax.axis('off')
    ax.set_aspect('equal')

    plt.xlim((-.05,1.05))
    plt.ylim((-.02, simplex.top_corner + 0.05))

    evolver = egt.analytical.StochDynamics(3, payoffs, Z)

    transition_matrix,fixation_probabilities = evolver.transition_and_fixation_matrix(5)
    stationary_distribution = egt.utils.calculate_stationary_distribution(transition_matrix)

    fig, bx = plt.subplots(figsize=(5, 5), dpi=150)
    G = egt.plotting.draw_stationary_distribution(strategy_labels,
                                                  1/Z, fixation_probabilities, stationary_distribution,
                                                  node_size=600, 
                                                  font_size_node_labels=8,
                                                  font_size_edge_labels=8,
                                                  font_size_sd_labels=8,
                                                  edge_width=1,
                                                  min_strategy_frequency=-0.01, 
                                                  ax=bx)
    plt.axis('off')
    plt.show()