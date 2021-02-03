from collections import defaultdict
import random


class Actor:

    def __init__(self, learning_rate, discount_factor, eli_decay, epsilon):
        self.epsilon = epsilon
        self.policy_dict = defaultdict(lambda: 0)
        self.eli_dict = {}  # defaultdict(lambda: 0)
        self.discount_factor = discount_factor
        self.eli_decay = eli_decay
        self.learning_rate = learning_rate

    def update_policy_dict(self, state, action, td_err):
        if (str(state), str(action)) in self.policy_dict.keys():
            self.policy_dict[(str(state), str(action))] += self.learning_rate * \
                td_err * self.get_elig(str(state), str(action))
        else:
            self.policy_dict[(str(state), str(action))] = self.learning_rate * \
                td_err * self.get_elig(state, action)

    # Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]

    def update_eli_dict(self, state, action, i):
        """
        Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]
        """
        if i == 0:
            self.eli_dict[(str(state), str(action))] = 1
            return
        else:
            value = self.get_elig(state, action) * \
                self.discount_factor * self.eli_decay
            element = {(str(state), str(action)): value}
            self.eli_dict.update(element)
            # self.eli_dict[(str(state), str(action))] = self.discount_factor * \
            #self.eli_decay * self.eli_dict[(str(state), str(action))]

    def get_elig(self, state, action):
        if (str(state), str(action)) in self.eli_dict.keys():
            return self.eli_dict[(str(state), str(action))]
        else:
            return 0

    def get_policy(self, state, action):
        """
        Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]
        """
        if (str(state), str(action)) in self.policy_dict.keys():
            return self.policy_dict[(str(state), str(action))]
        else:
            return 0

    def reset_eli_dict(self):
        self.eli_dict = {}  # defaultdict(lambda: 0)

    def get_action(self, state, legal_actions):
        """
        Returns action recommended by current policy, with the exception of random exploration epsilon percent of the time
        """
        self.epsilon = self.epsilon*0.9999
        if random.uniform(0, 1) >= self.epsilon:
            return max(legal_actions, key=lambda action: self.get_policy(state, action))
        return random.choice(legal_actions)


#actor = Actor(1, 1, 1, 1,0.999)


# actor.update_eli_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(2, 2), (2, 2)], 0)
# actor.update_eli_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(1, 1), (0, 0)], 0)
#
# actor.update_policy_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(2, 2), (2, 2)], 0.6)
# actor.update_policy_dict([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(1, 1), (0, 0)], 0.3)
# print(actor.policy_dict)
# print(actor.get_policy([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [(2, 2), (2, 2)]))
# print(actor.get_action([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], [[(2, 2), (2, 2)], [(1, 1), (0, 0)]]))

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
