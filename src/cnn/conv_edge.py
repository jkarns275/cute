import logging
from typing import List, Tuple, Optional, Dict, cast

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, make_batch_norm_layer
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.edge import Edge
if False:
    from cnn.layer import Layer


class ConvEdge(Edge):


    def __init__(self, edge_innovation_number: int, stride: int, input_layer_in: int, output_layer_in: int, layer_map: Dict[int, 'Layer'],
                       enabled: bool=True):
        self.stride: int = stride
       
        self.input_shape: Tuple[int, int, int] = layer_map[input_layer_in].output_shape
        self.output_shape: Tuple[int, int, int] = layer_map[output_layer_in].output_shape
        
        super().__init__(edge_innovation_number, self.input_shape, self.output_shape, input_layer_in, output_layer_in, layer_map, enabled)
        
        filter_width, filter_height = \
                calculate_required_filter_size(stride, *self.input_shape, *self.output_shape)
        self.filter_width: int = filter_width
        self.filter_height: int = filter_height
        self.number_filters: int = self.output_shape[2]

        # Make sure the filter size and number of filters we calculated are correct
        self.validate_output_volume_size()

        self.tf_layer: Optional[tf.Tensor] = None        


    def copy(self, layer_map: Dict[int, 'Layer']) -> 'ConvEdge':
        return ConvEdge(self.edge_innovation_number, self.stride, self.input_layer_in, self.output_layer_in, layer_map, self.enabled)


    def validate_output_volume_size(self):
        # Assume 0 padding.
        width, height, depth = calculate_output_volume_size(self.stride, 0, self.filter_width, self.filter_height, 
                                                            self.number_filters, *self.input_shape)
        
        assert (width, height, depth) == self.output_shape


    def validate_tf_layer_output_volume_size(self):
        assert self.tf_layer is not None

        shape = self.tf_layer.shape[1:]
        assert self.output_shape == shape


    def get_name(self):
        return f"conv_edge_inov_n_{self.edge_innovation_number}"
    

    def get_tf_layer(self, layer_map: Dict[int, 'Layer'], edge_map: Dict[int, Edge]) -> Optional[tf.Tensor]:
        if self.is_disabled():
            return None

        if self.tf_layer is not None:
            return self.tf_layer

        maybe_input_tf_layer: Optional[tf.Tensor] = \
                layer_map[self.input_layer_in].get_tf_layer(layer_map, edge_map)
        
        if maybe_input_tf_layer is None:
            return None
        
        input_tf_layer: tf.Tensor = cast(tf.Tensor, maybe_input_tf_layer)

        self.tf_layer = \
                keras.layers.Conv2D(self.number_filters, 
                                    (self.filter_width, self.filter_height), 
                                    strides=(self.stride, self.stride),
                                    activation='linear',
                                    input_shape=self.input_shape,
                                    name=self.get_name())(input_tf_layer)
        
        self.tf_layer = make_batch_norm_layer(name=self.get_name() + "_batch_norm")(self.tf_layer)
        self.tf_layer = make_activation_layer()(self.tf_layer)

        self.validate_tf_layer_output_volume_size()
        
        return self.tf_layer
