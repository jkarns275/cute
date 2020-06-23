import logging
from typing import List, Tuple, Optional

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.edge import Edge


class ConvEdge(Edge):


    def __init__(self, edge_innovation_number: int, stride: int, input_layer: 'Layer', output_layer: 'Layer'):
        self.stride: int = stride
       
        self.input_shape: Tuple[int, int, int] = input_layer.output_shape
        self.output_shape: Tuple[int, int, int] = output_layer.output_shape
        
        super().__init__(edge_innovation_number, self.input_shape, self.output_shape, input_layer, output_layer)
        
        filter_width, filter_height = \
                calculate_required_filter_size(stride, *self.input_shape, *self.output_shape)
        self.filter_width: int = filter_width
        self.filter_height: int = filter_height
        self.number_filters: int = self.output_shape[2]

        # Make sure the filter size and number of filters we calculated are correct
        self.validate_output_volume_size()

        self.tf_layer: Optional[tf.Tensor] = None        


    def validate_output_volume_size(self):
        # Assume 0 padding.
        width, height, depth = calculate_output_volume_size(self.stride, 0, self.filter_width, self.filter_height, 
                                                            self.number_filters, *self.input_shape)
        
        assert (width, height, depth) == self.output_shape


    def validate_tf_layer_output_volume_size(self):
        assert self.tf_layer is not None

        shape = self.tf_layer.shape[1:]
        assert self.output_layer.output_shape == shape


    def get_tf_layer(self) -> keras.layers.Layer:
        if self.tf_layer is not None:
            return self.tf_layer

        input_tf_layer: tf.Tensor = self.input_layer.get_tf_layer()
        self.tf_layer = \
                keras.layers.Conv2D(self.number_filters, 
                                    (self.filter_width, self.filter_height), 
                                    strides=(self.stride, self.stride),
                                    activation='linear',
                                    input_shape=self.input_shape)(input_tf_layer)
        
        self.tf_layer = make_activation_layer()(self.tf_layer)

        self.validate_tf_layer_output_volume_size()
        
        return self.tf_layer
