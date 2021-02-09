import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from split_gd import SplitGD


class CriticNN:
    def __init__(self, dims, alpha, eli_decay, gamma, shape, size):
        self.dims = dims
        self.alpha = alpha
        self.eli_decay = eli_decay
        self.gamma = gamma
        self.model = self.gennet(dims, size, alpha)
        self.splitGD = SplitGD(self.model, 0)
        self.shape = shape
        self.size = size
        self.studied = []

    def train(self, cases, targets):
        x_train = []
        y_train = []
        for i in reversed(cases):
            x_train.append(self.convert_state(i))
        for j in reversed(targets):
            y_train.append(self.convert_state(j))
        self.model = self.splitGD.fit(x_train, y_train)

    def compute_target(self, reward, s_prime):
        if s_prime not in self.studied:
            self.studied.append(s_prime)
            return random.uniform(0, 1)
        else:
            s_p = self.convert_state(s_prime)
            return reward + self.gamma*self.model.predict(s_p)

    def compute_td_err(self, state, state_prime, reward):
        if state_prime not in self.studied:
            self.studied.append(state_prime)
            print(state)
            s = self.convert_state(state)
            print(s)
            return reward + self.gamma*random.uniform(0, 1) - self.model.predict(s)
        else:
            s = self.convert_state(state)
            s_p = self.convert_state(state_prime)
            return reward + self.gamma*self.model.predict(s_p) - self.model.predict(s)

    def convert_state(self, state):
        return np.concatenate([np.array(i) for i in state])

    def gennet(self, dims, size, alpha=0.01, opt='SGD', loss='MeanSquaredError()', activation="relu", last_activation="softmax"):
        model = keras.models.Sequential()
        opt = eval('keras.optimizers.' + opt)
        loss = eval('tf.keras.losses.' + loss)
        for layer in range(len(dims)-1):
            model.add(keras.layers.Dense(dims[layer], activation=activation))
        model.add(keras.layers.Dense(dims[-1], activation=last_activation))
        model.compile(optimizer=opt(lr=alpha), loss=loss)
        return model
