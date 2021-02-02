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

        # Computes TD error from formula: r + discount_factor*V(s') - V(s)

    def compute_td_err(self, current_state, next_state,
                       reward):  # Skal reward tas inn her, eller skal det tas inn ved initialisering?
        return reward + self.discount_factor * self.get_value(next_state) - self.get_value(current_state)

    # Updates dictionary with the formula: V(S) = V(s) + learning_rate*td_err*eligibility
    def update_value_dict(self, state, td_err):  # Oppdater for å ta inn SAP om nødvendig?
        if str(state) in self.value_dict.keys():
          self.value_dict[str(state)] += self.learning_rate * td_err * self.eli_dict[str(state)]
        else:
          self.value_dict[str(state)] = self.learning_rate * td_err * self.eli_dict[str(state)]

    # Updates eligibility dictionary using replacing traces: 1 if S = St, discount_factor*eli_decay*eli_dict[state]
    def update_eli_dict(self, state, td_err, i):  # i is index of state in history
        if i == 0:
            self.eli_dict[str(state)] = 1
        else:
            self.eli_dict[str(state)] = self.discount_factor * self.eli_decay * self.eli_dict[str(state)]

    # Return value for given state
    def get_value(self, state):
        if str(state) in self.value_dict.keys():
            return self.value_dict[str(state)]
        else:
            return random.randint(0, 1)

    # Reset eli_dict after episode ends
    def reset_eli_dict(self):
        self.eli_dict = {}
