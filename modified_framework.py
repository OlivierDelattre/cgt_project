import egttools as egt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#import seaborn as sns
from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex



Z  = 100         # Population size
N  = 4           # Group size
b  = 1           # Endowment (individual's money/funds/...)
c  = 0.1         # Amount of money individuals contribute
Mc = 5           # Minimum collective contribution
M  = 1           # OR Minimum number of cooperators
r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments


def theta(x): return 0 if x < 0 else 1


class CRDWithExecutor():

	#### -----------------------------------------------------------------------------------------------------------
	#### ---- Adapted from https://github.com/Socrats/EGTTools/blob/stable/src/egttools/games/informal_risk.py -----
	#### -----------------------------------------------------------------------------------------------------------
    
    def __init__(self, strategies: list, initial_endowment, group_size: int, cost: float, risk:float, 
                alpha: float, cooperation_threshold: int, enhancement_factor:float, pi_t: float, pi_e: float, n_e: int,
                incentive:tuple=('local', 'fixed')):

        self.N = group_size                     # Size of groups
        self.M = cooperation_threshold          # Number of cooperators needed for success
        
        self.c = cost                           # Cost of cooperating
        self.r = risk                           # Risk of losing everything if num cooperators is not reached
        self.b = initial_endowment              # Initial funds of each individual

        self.alpha = alpha                      # Positive/Negative incentive trade-off
        self.delta = enhancement_factor         # Enhancement_factor?

        self.pi_t = pi_t                        # Cost of contributing to sanction pool
        self.pi_e = pi_e                        # Support from Executors?
        self.n_e  = n_e
       
        self.strategies_    = strategies        # List of strategies
        self.nb_strategies_ = len(strategies)   # Number of strategies

        self.ic = np.random.randint(Z)
        self.ie = np.random.randint(Z-self.ic)
        self.id = Z - self.ie - self.ic

        # Indices of Cooperator, Defector and Executor in strategy list
        indices = np.argsort([s.type() for s in strategies])
        self.ci, self.di, self.ei = indices    

        assert(len(incentive) == 2)
        assert(incentive[0] in ['global', 'local'])
        assert(incentive[1] in ['fixed', 'flexible'])

        self.incentive  = incentive
        self.nb_states_ = egt.calculate_nb_states(self.nb_strategies_, self.nb_strategies_)
        self.payoffs_   = np.zeros(shape=(self.nb_strategies_, self.nb_strategies_), dtype=np.float64)
        
    
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
    			fd += ((math.comb(self.ic, jc) * math.comb(self.ie, je) * math.comb(Z-self.ic-self.ie-1, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.defector_fixed_incentives_payoff(jc, je)

    	return fd 


    def cooperator_average_payoffs_fixed(self):

    	fc = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fc += ((math.comb(self.ic-1, jc) * math.comb(self.ie, je) * math.comb(Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.cooperator_fixed_incentives_payoff(jc+1, je)

    	return fc


    def executor_average_payoffs_fixed(self):

    	fe = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fe += ((math.comb(self.ic, jc) * math.comb(self.ie-1, je) * math.comb(Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.executor_fixed_incentives_payoff(jc, je+1)

    	return fe



    # average payoffs flexible

    def defector_average_payoffs_flexible(self):

    	fd = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fd += ((math.comb(self.ic, jc) * math.comb(self.ie, je) * math.comb(Z-self.ic-self.ie-1, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.defector_flexible_incentives_payoff(jc, je)

    	return fd 


    def cooperator_average_payoffs_flexible(self):

    	fc = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fc += ((math.comb(self.ic-1, jc) * math.comb(self.ie, je) * math.comb(Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.cooperator_flexible_incentives_payoff(jc+1, je)

    	return fc


    def executor_average_payoffs_flexible(self):

    	fe = 0

    	for jc in range(self.N):
    		for je in range(self.N - jc):
    			fe += ((math.comb(self.ic, jc) * math.comb(self.ie-1, je) * math.comb(Z-self.ic-self.ie, self.N-1-jc-je)) / math.comb(Z-1, self.N-1)) * self.executor_flexible_incentives_payoff(jc, je+1)

    	return fe


    def calculate_payoffs(self) -> np.ndarray:  # Array of shape (nb_strategies, nb_states_)

        payoffs = np.zeros((self.nb_strategies_, self.nb_strategies_))
        nb_states_ = egt.calculate_nb_states(self.nb_strategies_, self.nb_strategies_)

        for i in range(self.nb_strategies_):
            group_composition = egt.sample_simplex(i, self.nb_strategies_, self.nb_strategies_)

            if self.incentive[1] == 'fixed':
                PI_C = self.cooperator_average_payoffs_fixed()
                PI_D = self.defector_average_payoffs_fixed()
                PI_E = self.executor_average_payoffs_fixed()
            
            else:
                PI_C = self.cooperator_average_payoffs_flexible()
                PI_D = self.defector_average_payoffs_flexible()
                PI_E = self.executor_average_payoffs_flexible()
            
            payoffs[0, i] = PI_C
            payoffs[1, i] = PI_D
            payoffs[2, i] = PI_E

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


strategy_labels = [
    'Coop',
    'Defe',
    'Exec',
]

c = 0.1
b = 1
pi_t = 0.03
pi_e = 0.3
alpha = 1

game = CRDWithExecutor(strategies=[Cooperator(c, b), Defector(c, b), Executor(c, b, pi_t, pi_e, alpha)],
                initial_endowment=b,
                group_size=4,
                cost=c,
                risk=0.2,
                alpha=alpha,
                cooperation_threshold=2,
                enhancement_factor=0.1,
                pi_t=pi_t,
                pi_e=pi_e,
                n_e=0.25,)

payoffs = game.calculate_payoffs()

strategy_labels = ["Cooperator", "Defector", "Executor"]

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
plt.ylim((-.1, simplex.top_corner + 0.05))
plt.show()
















































