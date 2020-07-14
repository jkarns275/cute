import logging
from typing import List, Tuple, Optional, Dict, cast

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, make_batch_norm_layer, get_batch_size
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.conv_edge import ConvEdge
from cnn.edge import Edge

if False:
    from cnn.layer import Layer


class FractionalMaxPoolingEdge(ConvEdge):


    def __init__(self, edge_innovation_number: int, input_layer_in: int, output_layer_in: int, layer_map: Dict[int, 'Layer'],
                       enabled: bool=True):
        super().__init__(edge_innovation_number, 0, input_layer_in, output_layer_in, layer_map, enabled)


    def copy(self, layer_map: Dict[int, 'Layer']) -> 'FractionalMaxPoolingEdge':
        return FractionalMaxPoolingEdge(self.edge_innovation_number, self.input_layer_in, self.output_layer_in, layer_map, self.enabled)
    

    def validate_output_volume_size(self):
        iw, ih, ic = self.input_shape
        ow, oh, oc = self.output_shape

        assert iw > ow
        assert ih > oh
        assert ic <= oc


    def validate_tf_layer_output_volume_size(self):
        assert self.tf_layer is not None
        shape = self.tf_layer.shape[1:]
        assert self.output_shape[:-1] == shape[:-1]

        assert shape[-1] <= self.output_shape[-1]


    def get_name(self):
        return f"fractional_max_pooling_inov_n_{self.edge_innovation_number}"
    

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
        
        def max_pool(x):
            # https://www.tensorflow.org/api_docs/python/tf/nn/fractional_max_pool
            # returns three tensors
            mp = tf.nn.fractional_max_pool(x, [1., self.input_shape[0] / self.output_shape[0], self.input_shape[1] / self.output_shape[1], 1.])
            return mp.output

        self.tf_layer = keras.layers.Lambda(lambda x: max_pool(x), name=self.get_name())(input_tf_layer)
        
        if self.input_shape[2] < self.output_shape[2]:
            # Add some zero channels at the end so output shape is as expected
            # We have input_shape[2] channels but need to get to output_shape[2] channels
            zero_channels = self.output_shape[2] - self.input_shape[2]
            padding = tf.zeros((get_batch_size(), self.output_shape[0], self.output_shape[1], zero_channels), name=f"{self.get_name()}_padding")
            self.tf_layer = keras.layers.Concatenate(axis=3, name=self.get_name() + "_padding_concat")([self.tf_layer, padding])
        
        # self.tf_layer = make_batch_norm_layer(name=self.get_name() + "_batch_norm")(self.tf_layer)
        # self.tf_layer = make_activation_layer()(self.tf_layer)

        self.validate_tf_layer_output_volume_size()
        
        return self.tf_layer
