import logging
from typing import List, Tuple, Optional

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.edge import Edge
from cnn.output_layer import OutputLayer

class DenseEdge(Edge):


    def __init__(self, edge_innovation_number: int, input_layer: 'ConvLayer', output_layer: 'OutputLayer'):
        self.input_shape: Tuple[int, int, int] = input_layer.output_shape
        self.output_shape: Tuple[int, int, int] = output_layer.output_shape
        
        super().__init__(   edge_innovation_number, self.input_shape, self.output_shape,
                            input_layer, output_layer)
        
        self.tf_layer: Optional[tf.Tensor] = None

        assert type(output_layer) == OutputLayer
        

    def validate_tf_layer_output_volume_size(self):
        assert self.tf_layer is not None

        shape = tf.shape(self.tf_layer)[1:]

        assert (width, height, depth) == shape


    def get_tf_layer(self) -> keras.layers.Layer:
        if self.tf_layer is not None:
            return self.tf_layer

        input_tf_layer: tf.Tensor = self.input_layer.get_tf_layer()

        width, height, depth = self.input_layer.output_shape
        
        flattened = keras.layers.Flatten()(input_tf_layer)
        shape = flattened.shape[1]

        assert shape == width * height * depth

        number_units: int = self.output_layer.get_first_layer_size()

        # It is important that there is no activation function here. The activation function
        # will be applied after an Add layer is applied in the OutputLayer
        dense = keras.layers.Dense(number_units, input_shape=(shape,), activation='linear')(flattened)
        
        self.tf_layer = dense

        return self.tf_layer
