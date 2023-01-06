import egttools as egt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import seaborn as sns

from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex#, plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots
from egttools.plotting.helpers    import (xy_to_barycentric_coordinates, barycentric_to_xy_coordinates, find_roots_in_discrete_barycentric_coordinates, calculate_stability)
from egttools.analytical.utils    import (find_roots, check_replicator_stability_pairwise_games)
from egttools.helpers.vectorized  import vectorized_replicator_equation, vectorized_barycentric_to_xy_coordinates


def theta(x): return 0 if x < 0 else 1


class CRDWithExecutor():

	#### -----------------------------------------------------------------------------------------------------------
	#### ---- Adapted from https://github.com/Socrats/EGTTools/blob/stable/src/egttools/games/informal_risk.py -----
	#### -----------------------------------------------------------------------------------------------------------
    
    def __init__(self, strategies: list, initial_endowment, population_size: int, group_size: int, cost: float, risk:float, 
                alpha: float, cooperation_threshold: int, enhancement_factor:float, pi_t: float, pi_e: float, n_e: int, mu: int, incentive:tuple=('local', 'fixed')):

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
        self.id = self.Z - self.ie - self.ic

        self.incentive  = incentive
        self.nb_states_ = egt.calculate_nb_states(self.N, self.nb_strategies_)
        self.payoffs_   = np.zeros(shape=(self.nb_strategies_, self.nb_states_), dtype=np.float64)

    def set_population_state(self, state):
        if isinstance(state, (np.ndarray, list)):
            self.id, self.ie, self.ic = state
        else:
            self.id, self.ie, self.ic = egt.sample_simplex(state, self.Z, self.nb_strategies_)
 
    def Delta(self, je):

        if self.incentive[0] == 'local':
            delta = theta(je - self.n_e)
        else:
            delta = theta(self.ie - self.n_e)

        return delta

    # base 

    def base_defector_payoff(self, jc):
        return self.b * theta(jc - self.M) + (1 - self.r)*self.b*(1 - theta(jc - self.M))

    def base_cooperator_payoff(self, jc):
        return self.base_defector_payoff(jc) - self.c

    # fixed incentive 

    def defector_fixed_incentives_payoff(self, jc, je):
        return self.base_defector_payoff(jc+je) - ((1 - self.alpha)*self.pi_e*self.Delta(je))

    def cooperator_fixed_incentives_payoff(self, jc, je):
        return self.base_defector_payoff(jc+je) + (self.alpha*self.pi_e*self.Delta(je)) - self.c

    def executor_fixed_incentives_payoff(self, jc, je):
        return self.cooperator_fixed_incentives_payoff(jc, je) - self.pi_t

    # flexible incentive 

    def defector_flexible_incentives_payoff(self, jc, je):
        if (self.N - jc - je) == 0:
            return 0
        else:
            return self.base_defector_payoff(jc+je) - (1 - self.alpha)*((self.pi_t * je * self.delta)/(self.N - jc - je))*self.Delta(je)
    
    def cooperator_flexible_incentives_payoff(self, jc, je):
        if je + jc == 0:
            return 0
        else:
            return self.base_defector_payoff(jc+je) + self.alpha *((self.pi_t * je * self.delta)/(jc + je))*self.Delta(je) - self.c

    def executor_flexible_incentives_payoff(self, jc, je):
        return self.cooperator_flexible_incentives_payoff(jc, je) - self.pi_t

    # average payoffs fixed

    def defector_average_payoffs_fixed(self):
        fd = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                fd += ((math.comb(ic, jc) * math.comb(ie, je) * math.comb(self.Z-ic-ie-1, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.defector_fixed_incentives_payoff(jc, je)
        return fd

    def cooperator_average_payoffs_fixed(self):
        fc = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                fc += ((math.comb(ic-1, jc) * math.comb(ie, je) * math.comb(self.Z-ic-ie, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.cooperator_fixed_incentives_payoff(jc+1, je)
        return fc

    def executor_average_payoffs_fixed(self):
        fe = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                fe += ((math.comb(ic, jc) * math.comb(ie-1, je) * math.comb(self.Z-ic-ie, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.executor_fixed_incentives_payoff(jc, je+1)
        return fe

    # average payoffs flexible

    def defector_average_payoffs_flexible(self):
        fd = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                fd += ((math.comb(ic, jc) * math.comb(ie, je) * math.comb(self.Z-ic-ie-1, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.defector_flexible_incentives_payoff(jc, je)
        return fd 

    def cooperator_average_payoffs_flexible(self):
        fc = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
    	    for je in range(self.N - jc):
                fc += ((math.comb(ic-1, jc) * math.comb(ie, je) * math.comb(self.Z-ic-ie, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.cooperator_flexible_incentives_payoff(jc+1, je)
        return fc

    def cooperator_average_reward_flexible(self):
        rew = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                rew += ((math.comb(ic - 1, jc) * math.comb(ie, je) * math.comb(self.Z - ic - ie,
                                                                              self.N - 1 - jc - je)) \
                       / math.comb(self.Z - 1, self.N - 1)) * self.cooperator_flexible_incentives_payoff(jc + 1, je)
        return rew

    def executor_average_payoffs_flexible(self):
        fe = 0
        id, ie, ic = int(self.id), int(self.ie), int(self.ic)
        for jc in range(self.N):
            for je in range(self.N - jc):
                fe += ((math.comb(ic, jc) * math.comb(ie-1, je) * math.comb(self.Z-ic-ie, self.N-1-jc-je)) \
                    / math.comb(self.Z-1, self.N-1)) * self.executor_flexible_incentives_payoff(jc, je+1)
        return fe

    # Get right payoff function depending on which incentive strategy is used
    def defector_average_payoffs(self):
        if self.incentive[1] == 'fixed':
            return self.defector_average_payoffs_fixed()
        else:
            return self.defector_average_payoffs_flexible()

    def cooperator_average_payoffs(self):
        if self.incentive[1] == 'fixed':
            return self.cooperator_average_payoffs_fixed()
        else:
            return self.cooperator_average_payoffs_flexible()

    def executor_average_payoffs(self):
        if self.incentive[1] == 'fixed':
            return self.executor_average_payoffs_fixed()
        else:
            return self.executor_average_payoffs_flexible()

    def calculate_payoffs(self) -> np.ndarray:
        
        nb_states_ = egt.calculate_nb_states(self.N, self.nb_strategies_)
        payoffs = np.zeros((self.nb_strategies_, nb_states_))
        
        for i in range(nb_states_):
            
            group_composition = egt.sample_simplex(i, self.N, self.nb_strategies_)
            jc = group_composition[2]
            je = group_composition[1]

            if self.incentive[1] == 'fixed':
                PI_D = self.defector_fixed_incentives_payoff(jc, je)
                PI_E = self.executor_fixed_incentives_payoff(jc, je)
                PI_C = self.cooperator_fixed_incentives_payoff(jc, je)

            else:
                
                PI_D = self.defector_flexible_incentives_payoff(jc, je)
                PI_E = self.executor_flexible_incentives_payoff(jc, je)
                PI_C = self.cooperator_flexible_incentives_payoff(jc, je)
           
            payoffs[0, i] = PI_D
            payoffs[1, i] = PI_E
            payoffs[2, i] = PI_C
        
        self.payoffs_ = payoffs

        return self.payoffs_


    def T(self, L, R, pop_idx):
        self.set_population_state(pop_idx)
        pop = egt.sample_simplex(pop_idx, self.Z, self.nb_strategies_)
        f_L = self.payoffs_[L]
        f_R = self.payoffs_[R]
        i_L = pop[L]
        i_R = pop[R]
        if(L == 0):
            f_L = self.defector_average_payoffs()
        elif(L == 1):
            f_L = self.executor_average_payoffs()
        else:
            f_L = self.cooperator_average_payoffs()
        if(R == 0):
            f_R = self.defector_average_payoffs()
        elif(R == 1):
            f_R = self.executor_average_payoffs()
        else:
            f_R = self.cooperator_average_payoffs()
        
        return (1 - self.mu)*(i_L/self.Z)*(i_R/(self.Z-1))*(1 + math.e**(-self.beta*(f_R - f_L)))**-1 + self.mu*(i_L/(2*self.Z))

    def TCplus(self):
        return self.T(1, 2) + self.T(0, 2)
    def TCmin(self):
        return self.T(2, 1) + self.T(2, 0)
    def TEplus(self):
        return self.T(2, 1) + self.T(0, 1)
    def TEmin(self):
        return self.T(1, 2) + self.T(1, 0)

    def aG(self, i):
        self.set_population_state(i)
        total = 0
        ic, ie = self.ic, self.ie
        for jc in range(self.N + 1):
            for je in range(self.N - jc + 1):        
                total += math.comb(ic, jc)*math.comb(ie, je)*math.comb(int(self.Z-ic-ie), self.N-jc-je)*theta(jc + je - self.M)
        total *= (1/math.comb(self.Z, self.N))
        return total

    def aI(self, i):
            self.set_population_state(i)
            total = 0
            ic, ie = self.ic, self.ie
            for jc in range(self.N + 1):
                for je in range(self.N - jc + 1):        
                    total += math.comb(ic, jc)*math.comb(ie, je)*math.comb(int(self.Z-ic-ie), self.N-jc-je)*self.Delta(je)
            total *= (1/math.comb(self.Z, self.N))
            return total

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
    b  = 1.           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.3         # Minimum collective contribution
    M  = 3.          # OR Minimum number of cooperators
    r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 1.
    alpha = 0.
    mu    = 1/Z
    beta = 5.
    enhancement_factor = 1.4

    transitory = 10**2      # num of steps before we start counting
    nb_generations = 10**4  # num of steps where we do count
    nb_runs = 10            # num of different runs we average over

    strategy_labels = ["Defector", "Executor", "Cooperator"]

    game = CRDWithExecutor(strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                    initial_endowment=b,
                    population_size=Z,
                    group_size=N,
                    cost=c,
                    risk=r,
                    alpha=alpha,
                    cooperation_threshold=M,
                    enhancement_factor=enhancement_factor,
                    pi_t=pi_t,
                    pi_e=pi_e,
                    n_e=n_e,
                    mu=mu)

    payoffs = game.calculate_payoffs()
