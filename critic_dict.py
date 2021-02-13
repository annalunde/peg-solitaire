import random
from collections import defaultdict
from critic import Critic


# Using a dictionary to store state values
class TableCritic(Critic):
    """
    CriticDict keeps track of the value and eligibility of each state using dictionaries.
    """

    def __init__(self, learning_rate, eli_decay, discount_factor, dims=None):
        """
        Initializes a CriticDict using a default value of 0 to allow accumulated values.
        :param learning_rate: float
        :param eli_decay: float
        :param discount_factor: float
        """
        super().__init__(learning_rate=learning_rate, eli_decay=eli_decay, discount_factor=discount_factor)
        self.value_dict = {}
        self.eli_dict = defaultdict(lambda: 0)

    def compute_td_err(self, current_state, next_state, reward):
        """
        TD error describes the difference between reward + next state and the current state.
        Formula: r + discount_factor*V(s') - V(s)
        :param current_state: list(list(int))
        :param next_state: list(list(int))
        :param reward: int
        """
        return reward + self.discount_factor * self.get_value(next_state) - self.get_value(current_state)

    # Updates dictionary with the formula: V(S) = V(s) + learning_rate*td_err*eligibility
    # Oppdater for å ta inn SAP om nødvendig?
    def train(self, state, td_error):
        """
        Updates value_dict using the formula: V(S) = V(s) + learning_rate*td_err*eligibility
        :param state: list[list[boolean]]
        :param td_error: float
        """
        if str(state) in self.value_dict.keys():
            self.value_dict[str(state)] += self.learning_rate * td_error * self.eli_dict[str(state)]
        else:
            self.value_dict[str(state)] = self.learning_rate * td_error * self.eli_dict[str(state)]

    # Updates eligibility dictionary using replacing traces: 1 if S = St, discount_factor*eli_decay*eli_dict[state]
    def update_eligs(self, state, i):  # i is index of state in history
        """
        Updates eligibility dictionary using replacing traces: 1 if S = St (that is, i=0),
        else decay by formula: discount_factor*eli_decay*eli_dict[state]
        :param state: list[list[int]]
        :param i: int (0 for current state)
        """
        if i == 0:
            self.eli_dict[str(state)] = 1
        else:
            self.eli_dict[str(state)] = self.discount_factor * self.eli_decay * self.eli_dict[str(state)]

    # Return value for given state
    def get_value(self, state):
        """
        Return value for given state
        If state has not been seen before, return random initial value between 0 and 1
        :param state: list[list[int]]
        """
        if str(state) in self.value_dict.keys():
            return self.value_dict[str(state)]
        else:
            return random.randint(0, 1)

    # Reset eli_dict after episode ends
    def reset_eli_dict(self):
        self.eli_dict = {}
