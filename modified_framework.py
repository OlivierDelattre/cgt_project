import egttools as egt
import numpy as np
import math


Z = 100         # Population size
N = 10          # Group size
b = 10          # Endowment (individual's money/funds/...)
c = 1           # Amount of money individuals contribute
Mc = 5          # Minimum collective contribution
M = 5           # OR Minimum number of cooperators
r = 0.5         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments


def theta(x): return 0 if x < 0 else 1


class CRDWithExecutor(egt.games.AbstractGame):

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
        
        # Indices of Cooperator, Defector and Executor in strategy list
        indices = np.argsort([s.type() for s in strategies])
		self.ic, self.id, self.ie = indices     

        assert(len(incentive) == 2)
        assert(incentive[0] in ['global', 'local'])
        assert(incentive[1] in ['fixed', 'flexible'])

        self.incentive  = incentive
        self.nb_states_ = egt.calculate_nb_states(self.group_size, self.nb_strategies_)
        self.payoffs_   = np.zeros(shape=(self.nb_strategies_, self.nb_states_), dtype=np.float64)
        
        self.calculate_payoffs() # This updates the array above


    def delta(self, je):

        if self.incentive[0] == 'local':
            delta = theta(je - self.ne)
        else:
            delta = theta(self.ie - self.ne)

        return delta


    # base 

    def base_defector_payoff(self, jc):
        return self.b * theta(jc - self.M) + (1 - self.r)*self.b*(1 - theta(jc - self.M))

    def base_cooperator_payoff(self, jc):
        return self.base_defector_payoff(jc) - self.c


    # fixed incentive 

    def defector_fixed_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) - (1 - self.alpha)*pi_e*self.delta(je)

    def cooperator_fixed_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) + self.alpha*self.pi_e*self.delta(je) - self.c

    def executor_fixed_incentives_payoff(self, jc, je):
    	return self.cooperator_fixed_incentives_payoff(jc, je) - self.pi_t

    
    # flexible incentive 

    def defector_flexible_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) - (1 - self.alpha)*((self.pi_t * je * self.delta_lettre)/(self.N - jc - je))*self.delta(je)

    def cooperator_flexible_incentives_payoff(self, jc, je):
    	return self.base_defector_payoff(jc, je) + alpha *((self.pi_t * je * self.delta_lettre)/(jc + je))*self.delta(je) - self.c

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

    













































