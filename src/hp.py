from typing import Dict

# This file contains all of the hyper parameters.
import tensorflow as tf
import numpy as np

from dataset import Dataset


PARAMETER_COUNT_PENALTY_WEIGHT: float = 1e-8

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


__CNN_TRAINING_N_EPOCHS = 1
def set_number_epochs(epochs: int):
    __CNN_TRAINING_N_EPOCHS = epochs

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


def get_crossover_accept_rate(n: int):
    """
    for the nth worst parent, calculate the proportion of time nodes and edges should be accepted
    from that genome during crossover
    """
    return 1 / n


## When performing crossover, what proportion of the time nodes and edges from the more fit parent should 
# be put in the resulting child genome.
more_fit_crossover_rate: float  = 1.0

## When performing crossover, what proportion of the time nodes and edges from the less fit parent should
# be put in the resulting child genome.
less_fit_crossover_rate: float  = 0.50

# So these rates aren't quite proportions: the sum of them is the denominator, and the proportion
# is the fraction of a given rate and that sum.

## How often the add edge mutation should be performed
add_edge_rate: float            = 1.0

## How often the enable edge mutation should be performed
enable_edge_rate: float         = 1.0

## How often the disable edge mutation should be performed
disable_edge_rate: float        = 1.0

## How often the split edge mutation should be performed
split_edge_rate: float          = 1.0

## How often the clone mutation should be performed
clone_rate: float               = 1.0

## How often the add layer mutation should be performed
add_layer_rate: float                = 1.0

rate_sum: float = add_edge_rate + enable_edge_rate + disable_edge_rate + clone_rate + add_layer_rate

mutation_rates: Dict[str, float] = {
    'add_edge': add_edge_rate / rate_sum,
    'enable_edge': enable_edge_rate / rate_sum,
    'disable_edge': disable_edge_rate / rate_sum,
    'add_layer': add_layer_rate / rate_sum,
    'clone': clone_rate / rate_sum
}

assert 0.9999 < sum(mutation_rates.values()) < 1.0001
