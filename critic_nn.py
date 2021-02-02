import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import SplitGD

class Critic_nn():
  def __init__(self, dims, alpha, el_decay, gamma, shape):
    self.alpha = alpha
    self.el_decay = el_decay
    self.gamma = gamma
    self.model = self.gennet(dims, alpha)
    self.splitGD = SplitGD(self.model)
    self.eligs = {}

def run_episode_critic(self,s_init):
  reset_eligs(self)
  self.splitGD.fit(s_init, random.uniform(0, 1)) ###Target: random value between 0 and 1
  cases = {str(s_init):self.splitGD.model.predict(s_init)}
  current_state = s_init
  while not env.end_state(current_state): #call environment to check if end state
      s_prime, r = env.doAction(s_t) # Do action a from state s, moving the system to state s’ and receiving reinforcement r.
      cases[str(s_prime)] = r + self.gamma*self.splitGD.model.predict(s_t) #add target to cases
      self.splitGD.fit(cases.keys(), cases.values())
      td_err = compute_td(self,step,s_prime, r)
      update_elig(self,step)
      run_current_episode(self, cases, td_err)
      
def reset_eligs(self):
  self.eligs = {}

def run_current_episode(self,cases, td_err):
  for count,step in enumerate(cases):
    if count == 0:
        self.splitGD.model.predict(step) = self.splitGD.model.predict(step) + self.alpha*td_err*self.eligs[step]
        decay_elig(self,step)

def compute_td(self, state, state_prime, reward):
  return reward + self.gamma*self.model.predict(state_prime) - self.model.predict(state)

def update_elig(self, state):
  self.eligs[state] = 1

def decay_elig(self,state):
  self.eligs[state] =self.gamma*self.el_decay*self.eligs[state]


def gennet(dims, alpha=0.01, opt='SGD', loss='MeanSquaredError()' , activation= "relu", last_activation = "softmax"):
  model = keras.models.Sequential()
  opt = eval('keras.optimizers.'+ opt)
  loss = eval('tf.keras.losses.'+ loss)
  #model.add(tf.keras.Input(shape=(16,shape))) #tror den skal automatisk klare å forstå input_size
  for layer in range(len(dims)-1):
    model.add(keras.layers.Dense(dims[layer], activation=activation))
  model.add(keras.layers.Dense(dims[-1], activation = last_activaiton))
  model.compile(optimizer=opt(lr=alpha),loss=loss)
  return model

import numpy as np
labels = np.array([1,2,3,4,5])

train_data_initial = np.array([[1,2,4],[2,4,8],[3,6,9],[4,8,16],[5,10,20]])

input_ = labels.reshape(5,1,1,1)

train_data = train_data_initial.reshape(5,3,1,1)


print(train_data)

print('----')

print(input_)

import math
import tensorflow as tf
import numpy as np

# ************** Split Gradient Descent (SplitGD) **********************************
# This "exposes" the gradients during gradient descent by breaking the call to "fit" into two calls: tape.gradient
# and optimizer.apply_gradients.  This enables intermediate modification of the gradients.  You can find many other
# examples of this concept online and in the (excellent) book "Hands-On Machine Learning with Scikit-Learn, Keras,
# and Tensorflow", 2nd edition, (Geron, 2019).

# This class serves as a wrapper around a keras model.  Then, instead of calling keras_model.fit, just call
# SplitGD.fit.
#
# WARNING.  In THEORY, you should be able to use this class by just subclassing it and writing your own code
#  for the "modify_gradients" method.  However, there are many practical issues involving versions of tensorflow, use
# of keras and the tensorflow backend, etc.  So the main purpose of this file is to show the basics of how you can
# split gradient descent into two parts using tf.GradientTape.  Many similar examples exist online, but, again, there
# is no guarantee that they will work seamlessly with your own code.


class SplitGD():

    def __init__(self, keras_model, td_error):
        self.model = keras_model
        self.td_error = td_error
        self.eligs = {}

    # Subclass this with something useful.
    def modify_gradients(self,tape):   
        gradients = tape*(1/(-2*td_error)) 

        #udpate eligs
        self.eligs += gradients

        #final gradients = elig*td_error
      return self.eligs * self.td_error

    # This returns a tensor of losses, OR the value of the averaged tensor.  Note: use .numpy() to get the
    # value of a tensor.
    def gen_loss(self,features,targets,avg=False):
        predictions = self.model(features)  # Feed-forward pass to produce outputs/predictions
        loss = self.model.loss(targets, predictions) # model.loss = the loss function
        return tf.reduce_mean(loss).numpy() if avg else loss

    def fit(self, features, targets, epochs=1, mbs=1,vfrac=0.1,verbosity=1,callbacks=[]):
        params = self.model.trainable_weights
        train_ins, train_targs, val_ins, val_targs = split_training_data(features,targets,vfrac=vfrac)
        for cb in callbacks:    cb.on_train_begin()
        for epoch in range(epochs):
            for cb in callbacks:    cb.on_epoch_begin(epoch)
            for _ in range(math.floor(len(train_ins) / mbs)):
                with tf.GradientTape() as tape:  # Read up on tf.GradientTape !!
                    feaset,tarset = gen_random_minibatch(train_ins,train_targs,mbs=mbs)
                    loss = self.gen_loss(feaset,tarset,avg=False)
                    gradients = tape.gradient(loss,params)
                    gradients = self.modify_gradients(gradients)
                    self.model.optimizer.apply_gradients(zip(gradients,params))
            if verbosity > 0:
                self.end_of_epoch_action(train_ins,train_targs,val_ins,val_targs,epoch,
                                         verbosity=verbosity,callbacks=callbacks)
        for cb in callbacks:    cb.on_train_end()


    # The call to model.evaluate does 2 things for a set of features and targets: 1) computes the loss, 2) applies
    # the model's "metric" (which may differ from the loss) to produce an "evaluation".  A typical metric is
    # "categorical_accuracy" = the fraction of outputs that are "correct", i.e. the highest output neuron
    # corresponds to the correct value.  For more metrics, read up on Keras.metrics.
    # Verbosity levels: 0 = no prints, 1 = only my own prints, 2 = my prints + TF prints (in call to model.evaluate

    def gen_evaluation(self,features,targets,avg=False,verbosity=0,callbacks=[]):
        loss, evaluation = self.model.evaluate(features,targets,callbacks=callbacks,
                                               batch_size=len(features), verbose=(1 if verbosity == 2 else 0))
        return evaluation, loss
        # return (tf.reduce_mean(evaluation).numpy() if avg else evaluation), loss

    def status_display(self, val, loss, verbosity = 1, mode='Train'):
        if verbosity > 0:
            print('{0} *** Loss: {1} Eval: {2}'.format(mode,loss,val), end=' ')

    def end_of_epoch_action(self, train_ins, train_targs, valid_ins, valid_targs, epoch,verbosity=1,callbacks=[]):
        print('\n Epoch: {0}'.format(epoch), end=' ')
        # Calculate Loss and Evaluation for entire training set
        val, loss = self.gen_evaluation(train_ins, train_targs,avg=True,verbosity=verbosity,callbacks=callbacks)
        self.status_display(val,loss, verbosity=verbosity, mode='Train')
        val2, loss2 = 0, 0
        if len(valid_ins) > 0: # Calculate Loss and Evaluation for entire Validation Set
            val2, loss2 = self.gen_evaluation(valid_ins, valid_targs, avg=True, verbosity=verbosity,callbacks=callbacks)
            self.status_display(val2,loss2, verbosity=verbosity, mode='Validation')
        self.update_callbacks(epoch, (loss, val, loss2, val2), callbacks)

    def update_callbacks(self, epoch, quad, callbacks=[]):
        cb_log = {"loss": quad[0], "metric": quad[1], "val_loss": quad[2], "val_metric": quad[3]}
        #cb_log = {"loss": quad[0], "val_loss": quad[2]}
        for cb in callbacks:    cb.on_epoch_end(epoch,cb_log)


# A few useful auxiliary functions

def gen_random_minibatch(inputs, targets, mbs=1):
    indices = np.random.randint(len(inputs), size=mbs)
    return inputs[indices], targets[indices]

# This returns: train_features, train_targets, validation_features, validation_targets
def split_training_data(inputs,targets,vfrac=0.1,mix=True):
    vc = round(vfrac * len(inputs))  # vfrac = validation_fraction
    # pairs = np.array(list(zip(inputs,targets)))
    if vfrac > 0:
        pairs = list(zip(inputs,targets))
        if mix: np.random.shuffle(pairs)
        vcases = pairs[0:vc]; tcases = pairs[vc:]
        return np.array([tc[0] for tc in tcases]), np.array([tc[1] for tc in tcases]),\
               np.array([vc[0] for vc in vcases]), np.array([vc[1] for vc in vcases])
        #  return tcases[:,0], tcases[:,1], vcases[:,0], vcases[:,1]  # Can't get this to work properly
    else:
        return inputs,targets,[],[]
