import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from split_gd import SplitGD


class CriticNN:
    def __init__(self, dims, alpha, eli_decay, gamma, shape):
        self.alpha = alpha
        self.eli_decay = eli_decay
        self.gamma = gamma
        self.model = self.gennet(dims, alpha)
        self.splitGD = SplitGD(self.model, 0)
        self.eligs = {}
        self.shape = shape

  # def run_episode_critic(self, s_init):
   #     reset_eligs(self)
        # Target: random value between 0 and 1
    #    self.splitGD.fit(s_init, random.uniform(0, 1))
     #   cases = {str(s_init): self.splitGD.model.predict(s_init)}
      #  current_state = s_init
       # # call environment to check if end state
        # while not env.end_state(current_state):
        #   # Do action a from state s, moving the system to state s’ and receiving reinforcement r.
        #  s_prime, r = env.doAction(s_t)
        # cases[str(s_prime)] = r + self.gamma * \
        #    self.splitGD.model.predict(s_t)  # add target to cases
        # self.splitGD.fit(cases.keys(), cases.values())
        # td_err = compute_td_err(self, step, s_prime, r)
        # update_elig(self, step)
        # run_current_episode(self, cases, td_err)

    def reset_eli_dict(self):
        self.eligs = {}

    # def run_current_episode(self, cases, td_err):
     #   for count, step in enumerate(cases):
      #      if count == 0:
       #         self.splitGD.model.predict(step) = self.splitGD.model.predict(step) + self.alpha*td_err*self.eligs[step]
        #        decay_elig(self, step)

    def compute_td_err(self, state, state_prime, reward):
        s = convert_state(state)
        s_p = convert_state(state_prime)
        return reward + self.gamma*self.model.predict(s_p)) - self.model.predict(s)

    # def train_model(self, td_err):

    def convert_state(self, state):
      return np.array(state)

    def update_eli_dict(self, state, i):
        if i == 0:
            self.eligs[str(state)]=1
        else:
            self.eligs[str(state)]=self.gamma * \
                self.eli_decay*self.eligs[str(state)]

    def gennet(self, dims, alpha = 0.01, opt = 'SGD', loss = 'MeanSquaredError()', activation = "relu", last_activation = "softmax"):
        model=keras.models.Sequential()
        opt=eval('keras.optimizers.' + opt)
        loss=eval('tf.keras.losses.' + loss)
        # model.add(tf.keras.Input(shape=(16,shape))) #tror den skal automatisk klare å forstå input_size
        for layer in range(len(dims)-1):
            model.add(keras.layers.Dense(dims[layer], activation=activation))
        model.add(keras.layers.Dense(dims[-1], activation=last_activation))
        model.compile(optimizer = opt(lr=alpha), loss = loss)
        return model
