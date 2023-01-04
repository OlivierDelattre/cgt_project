import egttools as egt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#import seaborn as sns


from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex
from egttools.plotting.helpers    import (xy_to_barycentric_coordinates, barycentric_to_xy_coordinates, find_roots_in_discrete_barycentric_coordinates, calculate_stability)
from egttools.analytical.utils    import (find_roots, check_replicator_stability_pairwise_games)
from egttools.helpers.vectorized  import vectorized_replicator_equation, vectorized_barycentric_to_xy_coordinates


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
        self.id = self.Z - self.ie - self.ic

        self.incentive  = incentive
        self.nb_states_ = egt.calculate_nb_states(self.N, self.nb_strategies_)
        self.payoffs_   = np.zeros(shape=(self.nb_strategies_, self.nb_states_), dtype=np.float64)


    def Delta(self, je):

        #delta = 0
        #elif self.ie > self.Z * 0.25

        if je > self.N * 0.25:
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
        return self.base_defector_payoff(jc+je) - (1 - self.alpha)*self.pi_e*self.Delta(je)

    def cooperator_fixed_incentives_payoff(self, jc, je):
        return self.base_defector_payoff(jc+je) + self.alpha*self.pi_e*self.Delta(je) - self.c

    def executor_fixed_incentives_payoff(self, jc, je):
        return self.cooperator_fixed_incentives_payoff(jc, je) - self.pi_t

    # flexible incentive 

    def defector_flexible_incentives_payoff(self, jc, je):
        # Division by 0 when jc + je = 4
        res = self.base_defector_payoff(jc+je) - (1 - self.alpha)*((self.pi_t * je * self.delta)/(self.N - jc - je))*self.Delta(je)
        #print("jc+e res", jc+je, res)
        return res
    
    def cooperator_flexible_incentives_payoff(self, jc, je):
        return self.base_defector_payoff(jc+je) + self.alpha *((self.pi_t * je * self.delta)/(jc + je))*self.Delta(je) - self.c

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

    def calculate_payoffs(self) -> np.ndarray:
        nb_states_ = egt.calculate_nb_states(self.N, self.nb_strategies_)
        payoffs = np.zeros((self.nb_strategies_, nb_states_))
        for i in range(nb_states_):
            group_composition = egt.sample_simplex(i, self.N, self.nb_strategies_)
            jc = group_composition[2]
            je = group_composition[1]
            if self.incentive == 'fixed':
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

#    def calculate_payoffs(self) -> np.ndarray:  # Array of shape (nb_strategies, nb_states_)
#        
#        payoffs = np.zeros((self.nb_strategies_, self.nb_strategies_))
#
#        for i in range(self.nb_strategies_):
#
#            if self.incentive == 'fixed':
#                PI_D = self.defector_average_payoffs_fixed()
#                PI_E = self.executor_average_payoffs_fixed()
#                PI_C = self.cooperator_average_payoffs_fixed()
#            
#            else:
#                PI_D = self.defector_average_payoffs_flexible()
#                PI_E = self.executor_average_payoffs_flexible()
#                PI_C = self.cooperator_average_payoffs_flexible()
#            
#            payoffs[0, i] = PI_D
#            payoffs[1, i] = PI_E
#            payoffs[2, i] = PI_C
#
#        self.payoffs_ = payoffs
#
#        return self.payoffs_

    def T(self, L, R):
        f_L = self.payoffs_[L]
        f_R = self.payoffs_[R]
        if(L == 0):
            i_L = self.id
            f_L = self.defector_average_payoffs_fixed()
        elif(L == 1):
            i_L = self.ie
            f_L = self.executor_average_payoffs_fixed()
        else:
            i_L = self.ic
            f_L = self.cooperator_average_payoffs_fixed()
        
        if(R == 0):
            i_R = self.id
            f_R = self.defector_average_payoffs_fixed()
        elif(R == 1):
            i_R = self.ie
            f_R = self.executor_average_payoffs_fixed()
        else:
            i_R = self.ic
            f_R = self.cooperator_average_payoffs_fixed()
        
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
        pop = egt.sample_simplex(i, self.Z, self.nb_strategies_)
        total = 0
        ic, ie = int(pop[2]), int(pop[1])
        #ic, ie = self.ic, self.ie
        for jc in range(self.N):
            for je in range(self.N - jc):        
                total += math.comb(ic, jc)*math.comb(ie, je)*math.comb(int(self.Z-ic-ie), self.N-jc-je)*theta(jc + je - self.M)
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
    beta = 5

    strategy_labels = ["Defector", "Executor", "Cooperator"]

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

    simplex = egt.plotting.Simplex2D(discrete=True, size=Z, nb_points=Z+1)
    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * Z).astype(np.int64)

    evolver = egt.analytical.StochDynamics(3, payoffs, Z)

    result = np.asarray([[evolver.full_gradient_selection(v_int[:, i, j], beta) for j in range(v_int.shape[2])] for i in range(v_int.shape[1])]).swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    calculate_gradients = lambda u: Z*evolver.full_gradient_selection(u, beta)
    roots = find_roots_in_discrete_barycentric_coordinates(calculate_gradients, Z, nb_interior_points=5151, atol=1e-1)
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]
    stability = calculate_stability(roots, calculate_gradients)

    evolver.mu = 1/Z
    sd = evolver.calculate_stationary_distribution(beta)

    fig, ax = plt.subplots(figsize=(15,10))

    plot = (simplex.add_axis(ax=ax) 
        .apply_simplex_boundaries_to_gradients(Ux, Uy)
        .draw_gradients(zorder=5)
        .add_colorbar()
        .draw_stationary_points(roots_xy, stability, zorder=11)
        .add_vertex_labels(strategy_labels)
        .draw_stationary_distribution(sd, vmax=0.0001, alpha=0.5, edgecolors='gray',cmap='binary', shading='gouraud', zorder=0)
        .draw_trajectory_from_roots(lambda u, t: Z*evolver.full_gradient_selection_without_mutation(u, beta),
            roots,
            stability,
            trajectory_length=30,
            linewidth=1,
            step=0.001,
            color='k', draw_arrow=True, arrowdirection='right', arrowsize=30, zorder=10, arrowstyle='fancy')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    )

    ax.axis('off')
    ax.set_aspect('equal')
    plt.xlim((-.05,1.05))
    plt.ylim((-.02, simplex.top_corner + 0.05))
    plt.show()

    #print(payoffs)
    #print(payoffs.shape)

    #fig, ax = plt.subplots(figsize=(10,8))

    #simplex, gradients, roots, roots_xy, stability = plot_replicator_dynamics_in_simplex(payoffs, ax=ax)

    #plot = (simplex.draw_triangle()
              # .draw_gradients(density=1)
              # .add_colorbar(label='gradient of selection')
              # .add_vertex_labels(strategy_labels, epsilon_bottom=0.12)
              # .draw_stationary_points(roots_xy, stability)
              # .draw_scatter_shadow(lambda u, t: egt.analytical.replicator_equation(u, payoffs), 100, color='gray', marker='.', s=0.1)
              #)

    #ax.axis('off')
    #ax.set_aspect('equal')
    #plt.xlim((-.05,1.05))
    #plt.ylim((-.02, simplex.top_corner + 0.05))
    #plt.show()