from collections import defaultdict
import random


class Actor:

    def __init__(self, learning_rate, discount_factor, eli_decay, epsilon=None):
        self.epsilon = epsilon if epsilon else 0
        self.policy_dict = defaultdict(lambda: 0)
        self.eli_dict = defaultdict(lambda: 0)
        self.discount_factor = discount_factor
        self.eli_decay = eli_decay
        self.learning_rate = learning_rate

    # Updates policy using: current_policy + learning_rate*td_err*eli_dict[state,action]
    def update_policy_dict(self, state, action, td_err):
        self.policy_dict[(str(state), str(action))] += self.learning_rate * td_err * self.eli_dict[
            (str(state), str(action))]

    # Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]
    def update_eli_dict(self, state, action, i):
        if i == 0:
            self.eli_dict[(str(state), str(action))] = 1
        else:
            self.eli_dict[(str(state), str(action))] = self.discount_factor * self.eli_decay * self.eli_dict[
                (str(state), str(action))]

    def get_policy(self, state, action, length):
        if (str(state), str(action)) in self.policy_dict.keys():
            return self.policy_dict[(str(state), str(action))]
        else:
            return random.choice([i for i in range(length)])

    def reset_eli_dict(self):
        self.eli_dict = defaultdict(lambda: 0)

    def get_action(self, state, legal_actions):
        self.epsilon = self.epsilon*0.9999
        if random.uniform(0, 1) >= self.epsilon:
            return max(legal_actions, key=lambda action: self.get_policy(state, action, length=len(legal_actions)))
        return random.choice(legal_actions)


# actor = Actor(0.2, 0.2, 0.2)
#
# actor.update_eli_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(2, 3), (2, 1)], 0)
#
# actor.update_policy_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(2, 3), (2, 1)], 0.3)
# print(actor.policy_dict)
# print(actor.get_policy([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [(2, 3), (2, 1)]))
# print(actor.get_action([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [[(2, 1), (1, 1)], [(2, 3), (2, 1)]]))

# main

# env = Environment()
# a = Actor()
# c = Critic()

# for i in learning100:
#  env.newboardgame()
#  current_state = env.getstate()
#  legal_actions = env.get_legal_actions(current_state)

#  action = a.get_best_action(current_state, legal_actions)
#  reward = env.perform(action)
#  path.append(env.getstate())

#  td = critic.compute_td_err(newstate, reward)
#  for i, state in enumerate(path):
#    update_eli_dict(state, i)

#  a.update(current_state, action, td)
