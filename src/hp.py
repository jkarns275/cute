# This file contains all of the hyper parameters.
import tensorflow as tf

from datasets import Dataset


__CNN_ACTIVATION_TYPE = lambda: tf.keras.layers.LeakyReLU(alpha=0.1)
def make_activation_layer():
    return __CNN_ACTIVATION_TYPE()


__CNN_CLASSIFICATION_TYPE = tf.keras.layers.Softmax
def make_classification_layer():
    return __CNN_CLASSIFICATION_TYPE()


__CNN_BATCH_NORM_TYPE = tf.keras.layers.BatchNormalization
def make_batch_norm_layer(name=None):
    return __CNN_BATCH_NORM_TYPE(name=name)


# this needs to be set
__DATASET = 0
def set_dataset(dataset):
    global __DATASET
    
    __DATASET = dataset
    assert type(dataset) == Dataset

def get_dataset():
    assert __DATASET != 0

    return __DATASET
