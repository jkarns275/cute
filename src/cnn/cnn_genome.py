import logging
from typing import List, Dict

from tensorflow import keras

from cnn.cnn_util import make_edge_map
from cnn import Edge
from cnn import ConvEdge
from cnn import DenseEdge
from cnn import Layer
from cnn import InputLayer
from cnn import OutputLayer

from hp import get_dataset


class CnnGenome:


    def __init__(self,  number_outputs: int, input_layer: InputLayer, output_layer: OutputLayer,
                        layer_map: Dict[int, Layer], conv_edges: List[ConvEdge], output_edges: List[DenseEdge]):
        layer_map[input_layer.layer_innovation_number] = input_layer
        layer_map[output_layer.layer_innovation_number] = output_layer

        self.conv_edges: List[ConvEdge] = conv_edges
        self.output_edges: List[DenseEdge] = output_edges
        self.edge_map: Dict[int, Edge] = make_edge_map(self.conv_edges + self.output_edges)

        self.number_outputs: int = number_outputs
        self.fully_connected_layers: List[int] = [1024, number_outputs]
        
        self.input_layer: InputLayer = input_layer
        self.output_layer: OutputLayer = output_layer
        self.layer_map: Dict[int, Layer] = layer_map

        self.fitness = 100000.0
    
    def create_model(self):
        input_layer = self.input_layer.get_tf_layer(self.layer_map, self.edge_map)
        output_layer = self.output_layer.get_tf_layer(self.layer_map, self.edge_map)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    

    def train(self):
        logging.debug("called unimplemented method 'CnnGenome::train'")

        dataset = get_dataset()

        # Construct tensorflow model
        model: keras.Model = self.create_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Train it for some set number of epochs
        history = model.fit(dataset.x_train, dataset.y_train, batch_size=128, epochs=2, validation_data=(dataset.x_test, dataset.y_test), verbose=0)

        # Check the fitness
        fitness = history.history['loss'][-1]

        # set the fitness
        self.fitness = fitness

        logging.info(f"Trained model and got fitness of {self.fitness}")
