import logging
from typing import List, Tuple, Optional, Dict

import tensorflow.keras as keras
import tensorflow as tf

from cnn.edge import Edge


class Layer:

    
    NEXT_LAYER_INNOVATION_NUMBER: int = 0


    @staticmethod
    def get_next_layer_innovation_number():
        number = Layer.NEXT_LAYER_INNOVATION_NUMBER
        Layer.NEXT_LAYER_INNOVATION_NUMBER = number + 1
        return number


    def __init__(self, layer_innovation_number: int, width: int, height: int, depth: int):
        self.layer_innovation_number: int = layer_innovation_number
        
        self.width: int = width
        self.height: int = height
        self.depth: int = depth

        self.output_shape: Tuple[int, int, int] = (width, height, depth)

        self.inputs: List[Edge] = []

        self.tf_layer: Optional[tf.Tensor] = None


    def __getstate__(self):
        # Prevent the tensorflow layer from being pickled
        state = self.__dict__.copy()
        del state['tf_layer']
        return state


    def __setstate__(self, state):
        # Not sure if this is necessary but just make  
        self.__dict__.update(state)
        self.tf_layer = None


    def get_tf_layer(self, layer_map: Dict[int, 'Layer'], edge_map: Dict[int, Edge]) -> keras.layers.Layer:
        if self.tf_layer is not None:
            return self.tf_layer

        input_layers: List[tf.Tensor] = list(map(lambda edge_in: edge_map[edge_in].get_tf_layer(layer_map, edge_map), self.inputs))
        self.validate_tf_inputs(input_layers)
        
        if len(input_layers) > 1:
            self.tf_layer = keras.layers.Average()(input_layers)
        else:
            self.tf_layer = input_layers[0]

        return self.tf_layer


    def add_input_edge(self, input_edge: Edge):
        self.inputs.append(input_edge.edge_innovation_number)
        self.validate_input_edge(input_edge)
        
        # Just make sure the computation graph hasn't been created yet.
        assert self.tf_layer is None
        

    def validate_input_edge(self, input_edge: Edge):
        shape = input_edge.output_shape
        assert shape == (self.width, self.height, self.depth)


    def validate_tf_inputs(self, tf_inputs: List[tf.Tensor]):
        for input in tf_inputs:
            assert input.shape[1:] == self.output_shape
