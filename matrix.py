import egttools as egt
import numpy as np

Z = 100
N = 4
c = 0.1
b = 1
r = 0.2
m = 1/Z
pi_t = 0.03
pi_e = 0.3

group_size_ = 5
nb_strategies_ = 3


def generate_payoffs():
    nb_states_ = egt.calculate_nb_states(group_size_, nb_strategies_)
    for i in range(nb_states_):
        group_composition = egt.sample_simplex(i, group_size_, nb_strategies_)