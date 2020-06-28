# This file contains all of the hyper parameters.
import tensorflow as tf
import numpy as np

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


__CNN_LAYER_VOLUME_DEPTHS = (4, 8, 16, 32, 64, 128, 256)
def get_random_volume_depth(rng: np.random.Generator):
    return __CNN_LAYER_VOLUME_DEPTHS[rng.integers(0, len(__CNN_LAYER_VOLUME_DEPTHS))]


# The higher this is the more vram that will be consumed
__CNN_TRAINING_BATCH_SIZE = 64
def get_batch_size():
    return __CNN_TRAINING_BATCH_SIZE


__CNN_TRAINING_N_EPOCHS = 3
def get_number_epochs():
    return __CNN_TRAINING_N_EPOCHS


# this needs to be set
__DATASET = 0
def set_dataset(dataset):
    global __DATASET
    
    __DATASET = dataset
    assert type(dataset) == Dataset

def get_dataset():
    assert __DATASET != 0

    return __DATASET
