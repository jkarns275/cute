import logging
from typing import List, Tuple, Optional, Dict, Set

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


    def __init__(self,  layer_innovation_number: int, width: int, height: int, depth: int,
                        inputs: Set[int]=set(), outputs: Set[int]=set(), enabled: bool=True):
        self.layer_innovation_number: int = layer_innovation_number

        self.enabled: bool = enabled
        
        self.width: int = width
        self.height: int = height
        self.depth: int = depth

        self.output_shape: Tuple[int, int, int] = (width, height, depth)

        # Input edge innovation numbers
        self.inputs: Set[int] = inputs.copy()
        # Output edge innovation numbers
        self.outputs: Set[int] = outputs.copy()

        self.tf_layer: Optional[tf.Tensor] = None


    def __getstate__(self):
        # Prevent the tensorflow layer from being pickled
        state = self.__dict__.copy()
        del state['tf_layer']
        return state


    def __setstate__(self, state):
        if 'enabled' not in state:
            state['enabled'] = True
        self.__dict__.update(state)
        self.tf_layer = None

    
    def copy(self) -> 'Layer':
        return Layer(   self.layer_innovation_number, self.width, self.height, self.depth,
                        inputs=self.inputs, outputs=self.outputs)


    def enable(self):
        self.set_enabled(True)

    
    def disable(self):
        self.set_enabled(False)


    def set_enabled(self, enabled: bool):
        """
        This method is to be overridden, not the enable or disable methods
        """
        self.enabled = enabled
    

    def is_enabled(self):
        return self.enabled


    def clear_io(self):
        self.inputs = set()
        self.outputs = set()


    def get_tf_layer(self, layer_map: Dict[int, 'Layer'], edge_map: Dict[int, Edge]) -> Optional[keras.layers.Layer]:
        if self.tf_layer is not None:
            return self.tf_layer
        
        if not self.enabled:
            return None
        
        maybe_input_layers: List[Optional[tf.Tensor]] = list(map(lambda edge_in: edge_map[edge_in].get_tf_layer(layer_map, edge_map), self.inputs))
        input_layers: List[tf.Tensor] = [x for x in maybe_input_layers if x is not None]
            
        # There are no inputs return None
        if not input_layers:
            return None

        self.validate_tf_inputs(input_layers)
        
        if len(input_layers) > 1:
            self.tf_layer = keras.layers.Average()(input_layers)
        else:
            self.tf_layer = input_layers[0]

        return self.tf_layer


    def add_input_edge(self, input_edge: Edge):
        if input_edge.edge_innovation_number in self.inputs:
            logging.info(f"tried to add input edge {input_edge.edge_innovation_number} to layer {self.layer_innovation_number} more than once")
            return

        self.inputs.add(input_edge.edge_innovation_number)
        self.validate_input_edge(input_edge)
        
        # Just make sure the computation graph hasn't been created yet.
        assert self.tf_layer is None
    

    def add_output_edge(self, output_edge: Edge):
        if output_edge.edge_innovation_number in self.outputs:
            logging.info(f"tried to add output edge {output_edge.edge_innovation_number} to layer {self.layer_innovation_number} more than once")
            return
        
        self.outputs.add(output_edge.edge_innovation_number)
        self.validate_output_edge(output_edge)


    def validate_input_edge(self, input_edge: Edge):
        shape = input_edge.output_shape
        assert shape == self.output_shape
    

    def validate_output_edge(self, output_edge: Edge):
        shape = output_edge.input_shape
        assert shape == self.output_shape


    def validate_tf_inputs(self, tf_inputs: List[tf.Tensor]):
        for input in tf_inputs:
            assert input.shape[1:-1] == self.output_shape[:-1]
            assert input.shape[-1] <= self.output_shape[-1]
