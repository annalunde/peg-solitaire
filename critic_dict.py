import random
from collections import defaultdict


# Using a dictionary to store state values
class CriticDict:
    def __init__(self, learning_rate, eli_decay, discount_factor):
        self.value_dict = defaultdict(lambda: 0)
        self.eli_dict = defaultdict(lambda: 0)
        self.learning_rate = learning_rate
        self.eli_decay = eli_decay
        self.discount_factor = discount_factor

    def compute_td_err(self, current_state, next_state, reward):
        """
        Computes TD error from formula: r + discount_factor*V(s') - V(s)
        """
        return reward + self.discount_factor * self.get_value(next_state) - self.get_value(current_state)

    def update_value_dict(self, state, td_err):
        """
        Updates dictionary with the formula: V(S) = V(s) + learning_rate*td_err*eligibility
        """
        if str(state) in self.value_dict.keys():
            self.value_dict[str(state)] += self.learning_rate * td_err * self.eli_dict[str(state)]
        else:
            self.value_dict[str(state)] = self.learning_rate * td_err * self.eli_dict[str(state)]

    def update_eli_dict(self, state, i):  # i is index of state in history
        """
        Updates eligibility dictionary using replacing traces: 1 if S = St (that is, i=0),
        else decay by discount factor and eli_decay discount_factor*eli_decay*eli_dict[state]
        """
        if i == 0:
            self.eli_dict[str(state)] = 1
        else:
            self.eli_dict[str(state)] = self.discount_factor * self.eli_decay * self.eli_dict[str(state)]

    def get_value(self, state):
        """
        Return value for given state
        If state has not been seen before, return random initial value between 0 and 0.1
        """
        if str(state) in self.value_dict.keys():
            return self.value_dict[str(state)]
        else:
            return random.uniform(0, 0.1)

    def reset_eli_dict(self):
        """
        Reset eli_dict after episode ends
        """
        self.eli_dict = defaultdict(lambda: 0)
