# This file contains all of the hyper parameters.
import tensorflow as tf


_CNN_ACTIVATION_TYPE = lambda: tf.keras.layers.LeakyReLU(alpha=0.1)
def make_activation_layer():
    return _CNN_ACTIVATION_TYPE()
