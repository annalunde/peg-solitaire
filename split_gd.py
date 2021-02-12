import math
import tensorflow as tf
import numpy as np
from collections import defaultdict


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
# of keras and the tensorflow backend, etc.  So t
# he main purpose of this file is to show the basics of how you can
# split gradient descent into two parts using tf.GradientTape.  Many similar examples exist online, but, again, there
# is no guarantee that they will work seamlessly with your own code.


class SplitGD:

    def __init__(self, keras_model, td_error, alpha, gamma, eli_decay):
        self.model = keras_model
        self.td_error = td_error
        self.eligs = np.array([])
        self.gamma = gamma
        self.alpha = alpha
        self.eli_decay = eli_decay

    def decay_eligibilites(self):
        self.eligs = np.multiply(self.eligs, self.gamma * self.eli_decay)

    def update_td_error(self, td_err):
        self.td_error = td_err

    def reset_eli_dict(self):
        self.eligs = np.array([])

    # Subclass this with something useful.
    def modify_gradients(self, gradients, td_error):
        tape = tf.clip_by_global_norm(gradients, 10)[0]
        # Tape contains tensors of either two or one dimension, need to keep shape intact:
        if len(self.eligs) == 0:
            self.eligs = tf.zeros(shape=np.shape(gradients), dtype=tf.float32)
        self.eligs = np.add(self.eligs, gradients)
        gradients = np.multiply(self.eligs, td_error[0][0])
        self.decay_eligibilites()
        return gradients



    def fit(self, feature, td_error, epochs=1, mbs=1, vfrac=0.1, verbosity=1, callbacks=[]):
        params = self.model.trainable_weights
        # for cb in callbacks:
        #     cb.on_train_begin()
        # for epoch in range(epochs):
        #     for cb in callbacks:
        #         cb.on_epoch_begin(epoch)
        #     for _ in range(1):
        with tf.GradientTape() as tape:
            prediction = self.model(feature)
            gradients = tape.gradient(prediction, params)
            gradients = self.modify_gradients(gradients, td_error)
            self.model.optimizer.apply_gradients(
                zip(gradients, params))
            # if verbosity > 0:
            # self.end_of_epoch_action(train_ins, train_targs, val_ins, val_targs, epoch,
            #  verbosity=verbosity, callbacks=callbacks)
        #for cb in callbacks:
        #    cb.on_train_end()

    # The call to model.evaluate does 2 things for a set of features and targets: 1) computes the loss, 2) applies
    # the model's "metric" (which may differ from the loss) to produce an "evaluation".  A typical metric is
    # "categorical_accuracy" = the fraction of outputs that are "correct", i.e. the highest output neuron
    # corresponds to the correct value.  For more metrics, read up on Keras.metrics.
    # Verbosity levels: 0 = no prints, 1 = only my own prints, 2 = my prints + TF prints (in call to model.evaluate

    def gen_evaluation(self, features, targets, avg=False, verbosity=0, callbacks=[]):
        loss, evaluation = self.model.evaluate(features, targets, callbacks=callbacks,
                                               batch_size=len(features), verbose=(1 if verbosity == 2 else 0))
        return evaluation, loss
        # return (tf.reduce_mean(evaluation).numpy() if avg else evaluation), loss

    def status_display(self, val, loss, verbosity=1, mode='Train'):
        if verbosity > 0:
            print('{0} *** Loss: {1} Eval: {2}'.format(mode, loss, val), end=' ')

    def end_of_epoch_action(self, train_ins, train_targs, valid_ins, valid_targs, epoch, verbosity=1, callbacks=[]):
        print('\n Epoch: {0}'.format(epoch), end=' ')
        # Calculate Loss and Evaluation for entire training set
        val, loss = self.gen_evaluation(
            train_ins, train_targs, avg=True, verbosity=verbosity, callbacks=callbacks)
        self.status_display(val, loss, verbosity=verbosity, mode='Train')
        val2, loss2 = 0, 0
        if len(valid_ins) > 0:  # Calculate Loss and Evaluation for entire Validation Set
            val2, loss2 = self.gen_evaluation(
                valid_ins, valid_targs, avg=True, verbosity=verbosity, callbacks=callbacks)
            self.status_display(
                val2, loss2, verbosity=verbosity, mode='Validation')
        self.update_callbacks(epoch, (loss, val, loss2, val2), callbacks)

    def update_callbacks(self, epoch, quad, callbacks=[]):
        cb_log = {"loss": quad[0], "metric": quad[1],
                  "val_loss": quad[2], "val_metric": quad[3]}
        # cb_log = {"loss": quad[0], "val_loss": quad[2]}
        for cb in callbacks:
            cb.on_epoch_end(epoch, cb_log)


# A few useful auxiliary functions

def gen_random_minibatch(inputs, targets, mbs=1):
    indices = np.random.randint(len(inputs), size=mbs)
    return inputs[indices], targets[indices]


# This returns: train_features, train_targets, validation_features, validation_targets


def split_training_data(inputs, targets, vfrac=0.1, mix=True):
    vc = round(vfrac * len(inputs))  # vfrac = validation_fraction
    # pairs = np.array(list(zip(inputs,targets)))
    if vfrac > 0:
        pairs = list(zip(inputs, targets))
        if mix:
            np.random.shuffle(pairs)
        vcases = pairs[0:vc]
        tcases = pairs[vc:]
        return np.array([tc[0] for tc in tcases]), np.array([tc[1] for tc in tcases]), \
               np.array([vc[0] for vc in vcases]), np.array([vc[1]
                                                             for vc in vcases])
        #  return tcases[:,0], tcases[:,1], vcases[:,0], vcases[:,1]  # Can't get this to work properly
    else:
        return inputs, targets, [], []
