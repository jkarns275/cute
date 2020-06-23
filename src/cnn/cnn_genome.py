import logging

from typing import List

from cnn import ConvEdge
from cnn import DenseEdge
from cnn import Layer
from cnn import InputLayer
from cnn import OutputLayer

class CnnGenome:


    def __init__(self,  number_outputs: int, input_layer: InputLayer, output_layer: OutputLayer,
                        conv_layers: List[Layer], conv_edges: List[ConvEdge], output_edges: List[DenseEdge]):
        
        self.conv_layers: List[Layer] = conv_layers
        self.conv_edges: List[ConvEdge] = conv_edges
        self.output_edges: List[DenseEdge] = output_edges

        self.number_outputs: int = number_outputs
        self.fully_connected_layers: List[int] = [1024, number_outputs]
        
        self.input_layer: InputLayer = input_layer
        self.output_layer: OutputLayer = output_layer

    
    def create_model(self):
        input_layer = self.input_layer.get_tf_layer()
        output_layer = self.output_layer.get_tf_layer()


    def train(self):
        logging.debug("called unimplemented method 'CnnGenome::train'")
        # raise NotImplementedError

        # Construct tensorflow model

        # Train it for some set number of epochs

        # Check the fitness

        # set the fitness
