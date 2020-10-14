import logging
from typing import List, Tuple, Optional, Dict, Set, cast

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, make_batch_norm_layer, get_regularizer, get_weight_initialization
from cnn.cnn_util import calculate_output_volume_size, calculate_required_filter_size
from cnn.edge import Edge
if False:
    from cnn.layer import Layer
    from cnn import CnnGenome

class ConvEdge(Edge):


    def __init__(self, edge_innovation_number: int, stride: int, input_layer_in: int, output_layer_in: int, layer_map: Dict[int, 'Layer'],
                       enabled: bool=True):
        self.stride: int = stride
       
        self.input_shape: Tuple[int, int, int] = layer_map[input_layer_in].output_shape
        self.output_shape: Tuple[int, int, int] = layer_map[output_layer_in].output_shape
        super().__init__(edge_innovation_number, self.input_shape, self.output_shape, input_layer_in, output_layer_in, layer_map, enabled)
        
        filter_width, filter_height = \
                calculate_required_filter_size(stride, *self.input_shape, *self.output_shape)
        assert filter_width > 0
        assert filter_height > 0
        
        self.filter_width: int = filter_width
        self.filter_height: int = filter_height
        self.number_filters: int = self.output_shape[2]

        self.tf_weight_names = {self.get_name(), self.get_name() + "_batch_norm"}

        # Make sure the filter size and number of filters we calculated are correct
        self.validate_output_volume_size()

        self.tf_layer: Optional[tf.Tensor] = None        


    def validate_output_volume_size(self):
        """
        Ensures that the given input size and filter size lead to the required output volume size.
        """
        width, height, depth = calculate_output_volume_size(self.stride, 0, self.filter_width, self.filter_height, 
                                                            self.number_filters, *self.input_shape)
        
        assert (width, height, depth) == self.output_shape


    def copy(self, layer_map: Dict[int, 'Layer']) -> 'ConvEdge':
        """
        Returns a copy of the this ConvEdge.
        """
        return ConvEdge(self.edge_innovation_number, self.stride, self.input_layer_in, self.output_layer_in, layer_map, self.enabled)


    def validate_tf_layer_output_volume_size(self):
        """
        Ensures that the output volume size as calculated by TensorFlow matches the calculated output we have done.
        """
        assert self.tf_layer is not None
        shape = self.tf_layer.shape[1:]
        assert self.output_shape == shape


    def get_name(self):
        """
        Returns a unique identifier for this ConvEdge (since edge IN is unique).
        """
        return f"conv_edge_inov_n_{self.edge_innovation_number}"
    

    def get_tf_layer(self, genome: 'CnnGenome') -> Optional[tf.Tensor]:
        """
        A description of how this method works to construct a complete TensorFlow computation graph can be found
        in the documentation for the CnnGenome::create_model.
        
        Returns None if this is disabled or if the input layer get_tf_layer returns None.
        Otherwise returns a Tensor representing a convolution followed by activation then batch norm.
        
        """
        if self.is_disabled():
            return None

        # We only want to create this object once
        if self.tf_layer is not None:
            return self.tf_layer

        maybe_input_tf_layer: Optional[tf.Tensor] = \
                genome.layer_map[self.input_layer_in].get_tf_layer(genome)
        
        if maybe_input_tf_layer is None:
            return None
        
        input_tf_layer: tf.Tensor = cast(tf.Tensor, maybe_input_tf_layer)

        self.tf_layer = \
                keras.layers.Conv2D(self.number_filters, 
                                    (self.filter_width, self.filter_height), 
                                    strides=(self.stride, self.stride),
                                    kernel_regularizer=get_regularizer(genome.hp),
                                    bias_regularizer=get_regularizer(genome.hp),
                                    kernel_initializer=get_weight_initialization(genome),
                                    activation='linear',
                                    input_shape=self.input_shape,
                                    name=self.get_name())(input_tf_layer)
        
        self.tf_layer = make_activation_layer()(self.tf_layer)
        self.tf_layer = make_batch_norm_layer(name=self.get_name() + "_batch_norm")(self.tf_layer)

        self.validate_tf_layer_output_volume_size()
        
        return self.tf_layer
