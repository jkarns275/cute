import logging
from typing import List, Tuple, Optional

import tensorflow.keras as keras
import tensorflow as tf

from cnn.conv_edge import ConvEdge
from cnn.layer import Layer


class InputLayer(Layer):


    def __init__(self, layer_innovation_number: int, width: int, height: int, number_channels: int):
        super().__init__(layer_innovation_number, width, height, number_channels)

        # Also referrable by self.depth
        self.number_channels: int = number_channels


    def add_input(self, input_layer: keras.layers.Layer):
        raise RuntimeError("You cannot add an input to the input layer!")


    def get_tf_layer(self) -> keras.layers.Layer:
        if self.tf_layer is not None:
            return self.tf_layer

        self.tf_layer = keras.Input(shape=(self.width, self.height, self.number_channels))

        return self.tf_layer
