from typing import Dict, Optional, List

# This file contains all of the hyper parameters.
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from dataset import Dataset
if False:
    from cnn import CnnGenome

class EvolvableHP:

    WEIGHT_MAX = 2.0

    def __init__(self, min, max, init_min, init_max, type=float):
        self.min = min
        self.max = max
        self.init_min = init_min
        self.init_max = init_max
        self.type = type
    

    def random_initialization(self, rng: np.random.Generator):
        weight = rng.random()
        diff = self.max - self.min
        return self.type(diff * weight + self.min)

   
    def simplex(self, best, parent_values: List, rng: np.random.Generator):
        assert len(parent_values) >= 1
        assert type(best) == self.type
        assert list(map(type, parent_values)) == ([self.type] * len(parent_values))

        centroid = sum(parent_values) / len(parent_values)
        gradient = best - centroid 
        weight = EvolvableHP.WEIGHT_MAX * rng.random() - EvolvableHP.WEIGHT_MAX / 2

        midpoint = best + centroid / 2

        value = self.type(midpoint + gradient * weight)
        value = min(value, self.max)
        value = max(value, self.min)
        return self.type(value)

# how many genomes to generate before we start actually doing HP crossover, before that we just use random HPs
HP_WARM_UP_GENOMES: int = 40
class EvolvableHPConfig:
    

    PARAMETERS = {
        'lr': EvolvableHP(1e-7, 0.1, 0.001, 0.01, float),
        'beta_1': EvolvableHP(0.1, 0.999999999, 0.85, 0.95, float),
        'beta_2': EvolvableHP(0.9, 0.9999999999, 0.98, 0.999, float),
        'epsilon': EvolvableHP(1e-9, 1e-5, 1e-8, 1e-6, float),
        'bp_iters': EvolvableHP(1, 10, 1, 15, int),
        'l2_weight': EvolvableHP(1e-8, 1, 1e-7, 1e-3, float)
    }


    __slots__ = ['lr', 'beta_1', 'beta_2', 'epsilon', 'bp_iters', 'l2_weight']


    def __init__(self, parents: List['EvolvableHPConfig'], rng: np.random.Generator):
        if not parents:
            for name, hp in EvolvableHPConfig.PARAMETERS.items():
                setattr(self, name, hp.random_initialization(rng))
        else:
            # Need 2 or more for the simplex
            assert len(parents) > 1
            print(f"Best: {parents[0]}")
            for name, hp in EvolvableHPConfig.PARAMETERS.items():
                best_hp = getattr(parents[0], name)
                other_hps = list(map(lambda parent: getattr(parent, name), parents[1:]))
                setattr(self, name, hp.simplex(best_hp, other_hps, rng)) 


    def __str__(self):
        s = "{ "
        for key in EvolvableHPConfig.PARAMETERS.keys():
            s += f"{key} = {getattr(self, key)}, "

        s += "}"
        return s

L2_REGULARIZATION_WEIGHT: float         = 1e-8
PARAMETER_COUNT_PENALTY_WEIGHT: float   = 1e-8

def get_regularizer(hp):
    return keras.regularizers.l2(hp.l2_weight)

__CNN_ACTIVATION_TYPE = lambda: tf.keras.layers.LeakyReLU(alpha=0.1)
def make_activation_layer():
    return __CNN_ACTIVATION_TYPE()


__CNN_CLASSIFICATION_TYPE = tf.keras.layers.Softmax
def make_classification_layer():
    return __CNN_CLASSIFICATION_TYPE()


__CNN_BATCH_NORM_TYPE = tf.keras.layers.BatchNormalization
def make_batch_norm_layer(name=None):
    return __CNN_BATCH_NORM_TYPE(name=name)


__CNN_LAYER_VOLUME_DEPTHS = (2, 8, 16, 24)
def get_random_volume_depth(rng: np.random.Generator):
    return __CNN_LAYER_VOLUME_DEPTHS[rng.integers(0, len(__CNN_LAYER_VOLUME_DEPTHS))]


# The higher this is the more vram that will be consumed
__CNN_TRAINING_BATCH_SIZE = 20
def get_batch_size():
    return __CNN_TRAINING_BATCH_SIZE


__CNN_TRAINING_N_EPOCHS = 1
def set_number_epochs(epochs: int):
    __CNN_TRAINING_N_EPOCHS = epochs

def get_number_epochs(hp):
    return hp.bp_iters
    # return __CNN_TRAINING_N_EPOCHS


# this needs to be set
__DATASET = 0
def set_dataset(dataset):
    global __DATASET
    
    __DATASET = dataset
    assert type(dataset) == Dataset
    assert len(dataset.x_test) % __CNN_TRAINING_BATCH_SIZE == 0

def get_dataset():
    assert __DATASET != 0

    return __DATASET


def get_crossover_accept_rate(n: int):
    """
    this is assuming 0 index
    for the nth worst parent, calculate the proportion of time nodes and edges should be accepted
    from that genome during crossover
    """
    return 1 / (2 ** n)


__WEIGHT_INITIALIZATION_NAME = None
def set_weight_initialization(strategy_name: str):
    global __WEIGHT_INITIALIZATION
    strategy_name = strategy_name.lower()
    if strategy_name not in {'glorot', 'xavier', 'kaiming', 'epi', 'epigenetic'}:
        raise Exception(f"Supplied weight initialization strategy is invalid, unknown strategy '{strategy_name}'")
    __WEIGHT_INITIALIZATION_NAME = strategy_name


def get_weight_initialization(cnn_genome: 'CnnGenome'):
    
    if strategy_name in {'glorot', 'xavier'}:
        weight_initialization = tf.keras.initializers.GlorotUniform()
    elif strategy_name == 'kaiming':
        weight_initialization = tf.keras.initializers.he_normal()
    elif strategy_name in {'epi', 'epigenetic'}:
        weight_initialization = Epigenetic(cnn_genome)

    return __WEIGHT_INITIALIZATION



# How often to perform crossover
intra_island_co_rate: float = 0.2
inter_island_co_rate: float = 0.05
co_rate: float = intra_island_co_rate + inter_island_co_rate

mutation_rate: float = 1 - co_rate

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

## How often the copy mutation should be performed
copy_rate: float                = 1.0

## How often the add layer mutation should be performed
add_layer_rate: float           = 1.0

## How often the enable layer mutation should be performed
enable_layer_rate: float        = 1.0

## How often the disable layer mutation should be performed
disable_layer_rate: float       = 1.0

mutation_rate_sum: float = add_edge_rate + enable_edge_rate + disable_edge_rate + copy_rate + add_layer_rate + enable_layer_rate + disable_layer_rate

add_edge_probability: float         = add_edge_rate / mutation_rate_sum
enable_edge_probability: float      = enable_edge_rate / mutation_rate_sum
disable_edge_probability: float     = disable_edge_rate / mutation_rate_sum
add_layer_probability: float        = add_layer_rate / mutation_rate_sum
enable_layer_probability: float     = enable_layer_rate / mutation_rate_sum
disable_layer_probability: float    = disable_layer_rate / mutation_rate_sum
copy_probability: float             = copy_rate / mutation_rate_sum


# Probabilities for choosing the type of edge in the add_edge mutation
add_conv_edge_probability: float    = 0.25
add_separable_conv_edge_probability: float \
                                    = 0.25
add_pooling_edge_probability: float = 0.25
add_dense_edge_probability: float   = 0.25

