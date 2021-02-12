import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from split_gd import SplitGD


class CriticNN:
    def __init__(self, dims, alpha, eli_decay, gamma):
        self.dims = dims
        self.alpha = alpha
        self.eli_decay = eli_decay
        self.gamma = gamma
        self.model = self.gennet(dims, alpha)
        self.splitGD = SplitGD(self.model, 0, self.alpha,
                               self.gamma, self.eli_decay)
        self.studied = []

    def train(self, state, td_error):
        x_train = self.convert_state_to_tensor(state)
        td_error_tensor = tf.reshape(td_error, [1, 1])
        #print("y_train", error2)
        self.model = self.splitGD.fit(feature=x_train, td_error=td_error_tensor)


    def compute_td_err(self, state, state_prime, reward):
        if state not in self.studied:
            self.studied.append(state)
            state_value = random.uniform(0, 1)
        else:
            s = self.convert_state_to_tensor(state)
            state_value = self.splitGD.model.predict(s)[0][0]

        if state_prime not in self.studied:
            state_prime_value = random.uniform(0, 1)
        else:
            s_p = self.convert_state_to_tensor(state_prime)
            state_prime_value = self.splitGD.model.predict(s_p)[0][0]
        return reward + self.gamma*state_prime_value - state_value

    def convert_state_to_tensor(self, state):
        tensor = tf.convert_to_tensor(
            [np.concatenate([np.array(i) for i in state])])
        return tf.reshape(tensor, [1, 16])

    def gennet(self, dims, alpha=0.01, opt='SGD', loss='MeanSquaredError()', activation="relu", last_activation="relu"):
        model = keras.models.Sequential()
        opt = eval('keras.optimizers.' + opt)
        loss = eval('tf.keras.losses.' + loss)
        model.add(keras.layers.Dense(input_shape=(dims[0],),  # Determines shape after first input of a board state
                                     units=dims[0], activation=activation))
        for layer in range(1, len(dims)-1):
            model.add(keras.layers.Dense(
                units=dims[layer], activation=activation))
        model.add(keras.layers.Dense(
            dims[-1], activation=last_activation))
        model.compile(optimizer=opt(lr=alpha), loss=loss)
        return model
