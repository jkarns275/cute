import logging
from typing import List, Tuple, Optional, Dict, cast

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, get_regularizer
from cnn.edge import Edge
from cnn.layer import Layer
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.output_layer import OutputLayer


class DenseEdge(Edge):


    def __init__(self,  edge_innovation_number: int, input_layer_in: int, output_layer_in: int, layer_map: Dict[int, Layer],
                        enabled: bool=True):
        self.input_shape: Tuple[int, int, int] = layer_map[input_layer_in].output_shape
        self.output_shape: Tuple[int, int, int] = layer_map[output_layer_in].output_shape
        
        super().__init__(   edge_innovation_number, self.input_shape, self.output_shape,
                            input_layer_in, output_layer_in, layer_map, enabled)
        
        self.tf_layer: Optional[tf.Tensor] = None
        assert type(layer_map[output_layer_in]) == OutputLayer
       

    def copy(self, layer_map: Dict[int, Layer]) -> 'DenseEdge':
        return DenseEdge(self.edge_innovation_number, self.input_layer_in, self.output_layer_in, layer_map, self.enabled)


    def validate_tf_layer_output_volume_size(self):
        assert self.tf_layer is not None

        shape = tf.shape(self.tf_layer)[1:]

        assert (width, height, depth) == shape


    def get_name(self):
        return f"dense_edge_inov_n_{self.edge_innovation_number}"


    def get_tf_layer(self, layer_map: Dict[int, Layer], edge_map: Dict[int, Edge]) -> keras.layers.Layer:
        if self.is_disabled():
            return None

        if self.tf_layer is not None:
            return self.tf_layer
         
        input_layer: Layer = layer_map[self.input_layer_in]
        output_layer: OutputLayer = cast(OutputLayer, layer_map[self.output_layer_in])

        assert type(output_layer) == OutputLayer

        maybe_input_tf_layer: Optional[tf.Tensor] = input_layer.get_tf_layer(layer_map, edge_map)
        
        if maybe_input_tf_layer is None:
            return None

        input_tf_layer: tf.Tensor = cast(tf.Tensor, maybe_input_tf_layer)

        width, height, depth = input_layer.output_shape
        
        flattened = keras.layers.Flatten()(input_tf_layer)
        shape = flattened.shape[1]

        assert shape == width * height * depth

        number_units: int = output_layer.get_first_layer_size()

        # It is important that there is no activation function here. The activation function
        # will be applied after an Add layer is applied in the OutputLayer
        dense = \
            keras.layers.Dense( number_units, 
                                input_shape=(shape,),
                                activation='linear',
                                kernel_regularizer=get_regularizer(),
                                bias_regularizer=get_regularizer(),
                                name=self.get_name())(flattened)
        
        self.tf_layer = dense

        return self.tf_layer
