import logging
from typing import List, Tuple, Optional

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer
from cnn.conv_edge import ConvEdge
from cnn.layer import Layer


class OutputLayer(Layer):
    """
    This is a subclass of Layer so that the add_input, __getstate__, and __setstate__ methods are inherited.
    Also inherited is the 'inputs' field, which is a list of the input layers
    """


    def __init__(self, layer_innovation_number: int, dense_layers: List[int], number_classes: int):
        """
        Parameters
        ----------
        dense_layers    : :obj:`list` of :obj:`int`
            A list of fully connected layer sizes which will be placed at the end of the network. 
            If this list is empty the only fully connected layer will be the softmax layer.
        number_classes  : int
            The number of possible output classes. This means the final layer in the network
            will have `number_classes` nodes.
        """
        super().__init__(layer_innovation_number, number_classes, 1, 1)
        
        self.layer_innovation_number: int = layer_innovation_number
        self.number_classes: int = number_classes
        self.dense_layers: List[int] = dense_layers + [number_classes]

    
    def get_first_layer_size(self):
        return self.dense_layers[0]


    def get_tf_layer(self) -> keras.layers.Layer:
        if self.tf_layer is not None:
            return self.tf_layer
        
        number_units = self.dense_layers[0]

        # To make it easy to inherit epigenetic weights, we seperate the weights out into separate
        # layers without activation functions, and add the resulting layers together and then apply
        # an activation function.

        intermediate_layers = list(map(lambda layer: layer.get_tf_layer(), self.inputs))

        if len(intermediate_layers) > 1:
            sum_layer = keras.layers.Add()(intermediate_layers)
        else:
            sum_layer = intermediate_layers[0]

        activation_layer: tf.Tensor = make_activation_layer()(sum_layer)
        
        layer: tf.Tensor = activation_layer

        for size in self.dense_layers[1:]:
            shape = layer.shape[1:]
            layer = keras.layers.Dense(size, input_shape=shape, activation='linear')(layer)
            layer = make_activation_layer()(layer)
        
        self.tf_layer = layer

        return self.tf_layer
