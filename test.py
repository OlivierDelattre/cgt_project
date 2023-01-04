import egttools as egt
import numpy as np


class test1:
    def __init__(self):
        self.t = "1"
        self.a, self.b, self.c = [1, 2, 3]

    def test(self, a:tuple):
        return a

    def type(self):
        return self.t

        
class test2:
    def __init__(self):
        self.t = "2"

    def type(self):
        return self.t

        
class test3:
    def __init__(self):
        self.t = "3"

    def type(self):
        return self.t

strategies = [
    test1(),
    test3(),
    test2(),
]

strategies_ = np.array(strategies)[np.argsort([s.type() for s in strategies])]

group_size_ = 5
nb_strategies_ = 3

nb_states_ = egt.calculate_nb_states(group_size_, nb_strategies_)
for i in range(nb_states_):
    group_composition = egt.sample_simplex(i, group_size_, nb_strategies_)


for i in range(group_size_+1):
    state = [i, group_size_-i, 0]
    print(egt.calculate_state(group_size_, state))