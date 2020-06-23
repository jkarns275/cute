import logging
from typing import List, Tuple, Optional

import tensorflow.keras as keras
import tensorflow as tf


class Edge:


    NEXT_EDGE_INNOVATION_NUMBER: int = 0


    @staticmethod
    def get_next_layer_innovation_number():
        number = ConvEdge.NEXT_EDGE_INNOVATION_NUMBER
        ConvEdge.NEXT_EDGE_INNOVATION_NUMBER = number + 1
        return number


    def __init__(self, edge_innovation_number: int, input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int], input_layer: 'Layer', output_layer: 'Layer'):
        self.edge_innovation_number: int = edge_innovation_number
        self.input_layer: 'Layer' = input_layer
        self.output_layer: 'Layer' = output_layer
       
        self.input_shape: Tuple[int, int, int] = input_shape
        self.output_shape: Tuple[int, int, int] = output_shape
        
        self.tf_layer: Optional[tf.Tensor] = None
        
        output_layer.add_input(self)

    def __getstate__(self):
        # Prevent the tensorflow layer from being pickled
        state = self.__dict__.copy()
        del state['tf_layer']


    def __setstate__(self, state):
        # Not sure if this is necessary but just make  
        self.__dict__.update(state)
        self.tf_layer = None


    def get_tf_layer(self) -> keras.layers.Layer:
        raise NotImplementedError("Call to abstract method Edge::get_tf_layer")
