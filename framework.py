import egttools as egt
import numpy as np

Z = 100         # Population size
N = 10          # Group size
b = 10          # Endowment (individual's money/funds/...)
c = 1           # Amount of money individuals contribute
Mc = 5          # Minimum collective contribution
M = 5           # OR Minimum number of cooperators
r = 0.5         # If minimum is not met: All group participants lose their endowment
                #   with probability r
                # If minimum is met: individuals retain their endowments

def theta(x): return 0 if x < 0 else 1

#### -----------------------------------------------------------------------------------------------------------
#### ---- Adapted from https://github.com/Socrats/EGTTools/blob/stable/src/egttools/games/informal_risk.py -----
#### -----------------------------------------------------------------------------------------------------------
class CRDWithExecutor(egt.games.AbstractGame):
    def __init__(self, strategies: list, initial_endowment, group_size: int, cost: float, risk:float, 
                alpha: float, cooperation_threshold: int, enhancement_factor:float, pi_t, pi_e, 
                incentive:tuple=('local', 'fixed')):
        self.N = group_size
        self.c = cost
        self.r = risk
        self.alpha = alpha
        self.b = initial_endowment
        self.M = cooperation_threshold
        self.pi_t = pi_t
        self.pi_e = pi_e
        self.delta = enhancement_factor

        self.strategies_ = strategies
        self.nb_strategies_ = len(strategies)
        indices = np.argsort([s.type() for s in strategies])
        self.ci, self.di, self.ei = indices # Indices of Cooperator, Defector and Executor
        
        assert(len(incentive) == 2)
        assert(incentive[0] in ['global', 'local'])
        assert(incentive[1] in ['fixed', 'flexible'])

        self.incentive = incentive

        self.nb_states_ = egt.calculate_nb_states(self.group_size, self.nb_strategies_)
        self.payoffs_ = np.zeros(shape=(self.nb_strategies_, self.nb_states_), dtype=np.float64)
        self.calculate_payoffs() # This updates the array above

    
    # PI'D(jc) on page 11
    def base_defector_payoff(self, jc):
        return self.b * theta(jc - self.M) +\
                (1 - self.r)*self.b*(1 - theta(jc - self.M))

    def DELTA(self, group_composition:list):
        if self.incentive[0] == 'local':
            delta = 0 # TODO implement
            raise NotImplementedError
        else:
            delta = 0 # TODO implement
            raise NotImplementedError
        return delta

    # Fixed incentive payoff page 12
    def fixed_incentive_payoffs(self, group_composition: list):
        jc = group_composition[self.ci]
        jd = group_composition[self.di]
        je = group_composition[self.ei]
        PI_prime_D = self.base_defector_payoff(jc+je)

        PI_D = PI_prime_D - (1 - self.alpha)*self.pi_e*self.DELTA(group_composition)
        PI_C = PI_prime_D + self.alpha*self.pi_e*self.DELTA(group_composition) - self.c
        PI_E = PI_prime_D + self.alpha*self.pi_e*self.DELTA(group_composition) - self.c - self.pi_t

        return (PI_C, PI_D, PI_E)

    # Flexible incentive payoff page 12
    def flexible_incentive_payoffs(self, group_composition: list):
        jc = group_composition[self.ci]
        jd = group_composition[self.di]
        je = group_composition[self.ei]
        PI_prime_D = self.base_defector_payoff(jc+je)

        PI_D = PI_prime_D - (1 - self.alpha)*((self.pi_t*self.delta*je)/(self.N - jc - je))*self.DELTA(group_composition)
        PI_C = PI_prime_D + self.alpha*((self.pi_t*self.delta*je)/(jc + je))*self.DELTA(group_composition) - self.c
        PI_E = PI_prime_D + self.alpha*((self.pi_t*self.delta*je)/(jc + je))*self.DELTA(group_composition) - self.c - self.pi_t

        return (PI_C, PI_D, PI_E)
        
    def play(self, group_composition: list(int), game_payoffs: list) -> None:
        jc = group_composition[self.ci]
        jd = group_composition[self.di]
        je = group_composition[self.ei]
        
        # Flexible incentives (page 12)
        pi_D = 
        # Fixed incentives (page 12)

    def calculate_payoffs(self) -> np.ndarray:
        raise NotImplementedError

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: np.ndarray) -> float:
        raise NotImplementedError

    def payoffs(self)->np.ndarray:
        return self.payoffs_

    def payoff(self, strategy: int, group_composition: list)->float:
        raise NotImplementedError

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

class Cooperator(egt.behaviors.CRD.AbstractCRDStrategy):
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return self.c

    def type(self):
        return "Cooperator"

class Defector(egt.behaviors.CRD.AbstractCRDStrategy):
    def __init__(self, c, b):
        self.c = c
        self.endowment = b
        return

    def get_action(self, time_step:int, group_contributions_prev:int):
        return 0

    def type(self):
        return "Defector"

class Executor(egt.behaviors.CRD.AbstractCRDStrategy):
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

strategies = [
    egt.behaviors.NormalForm.CRD.TwoActions.Defector,
    egt.behaviors.NormalForm.TwoActions.Cooperator,
]