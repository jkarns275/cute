import logging
from typing import List, Tuple, Optional, Dict, cast

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, make_batch_norm_layer
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.conv_edge import ConvEdge
from cnn.edge import Edge

if False:
    from cnn.layer import Layer


class SeparableConvEdge(ConvEdge):


    def __init__(self, edge_innovation_number: int, stride: int, input_layer_in: int, output_layer_in: int, layer_map: Dict[int, 'Layer'],
                       enabled: bool=True):
        super().__init__(edge_innovation_number, stride, input_layer_in, output_layer_in, layer_map, enabled)


    def copy(self, layer_map: Dict[int, 'Layer']) -> 'SeparableConvEdge':
        return SeparableConvEdge(self.edge_innovation_number, self.stride, self.input_layer_in, self.output_layer_in, layer_map, self.enabled)


    def get_name(self):
        return f"factorized_conv_edge_inov_n_{self.edge_innovation_number}"
    

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
        
        assert self.filter_width == self.filter_height
        self.tf_layer = keras.layers.SeparableConv2D(   self.number_filters,
                                                        (self.filter_width, self.filter_height),
                                                        strides=(self.stride, self.stride),
                                                        activation='linear',
                                                        input_shape=self.input_shape,
                                                        name=self.get_name() + "_" + str(self.filter_height) + "x1")(input_tf_layer)
        
        self.tf_layer = make_batch_norm_layer(name=self.get_name() + "_batch_norm")(self.tf_layer)
        self.tf_layer = make_activation_layer()(self.tf_layer)

        self.validate_tf_layer_output_volume_size()
        
        return self.tf_layer
