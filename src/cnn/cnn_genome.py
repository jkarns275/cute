import logging
from typing import List, Dict, Any, cast

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
                        layer_map: Dict[int, Layer], conv_edges: List[ConvEdge], output_edges: List[DenseEdge],
                        epigenetic_weights: Dict[str, Any]={}, fitness: float=float('inf')):
        # When this object is serialized the input and output layer in this map are copied so we need to make sure
        # we use the same object, otherwise layer_map[input_layer.layer_innovation_number] and input_layer will be
        # equal but different objects.
        layer_map[input_layer.layer_innovation_number] = input_layer
        layer_map[output_layer.layer_innovation_number] = output_layer

        self.conv_edges: List[ConvEdge] = conv_edges
        for conv_edge in conv_edges:
            assert type(conv_edge) == ConvEdge

        self.output_edges: List[DenseEdge] = output_edges
        for output_edge in output_edges:
            assert type(output_edge) == DenseEdge

        self.edge_map: Dict[int, Edge] = make_edge_map(cast(List[Edge], self.conv_edges) + cast(List[Edge], self.output_edges))

        self.number_outputs: int = number_outputs
        self.fully_connected_layers: List[int] = [1024, number_outputs]
        
        self.input_layer: InputLayer = input_layer
        self.output_layer: OutputLayer = output_layer
        self.layer_map: Dict[int, Layer] = layer_map

        self.fitness: float = fitness

        # Not sure what type the weights will be
        self.epigenetic_weights: Dict[str, Any] = epigenetic_weights
  
        self.island: int = -1


    def copy(self) -> 'CnnGenome':
        copy_edge = lambda edge: edge.copy(self.layer_map)
        
        conv_edges = list(map(copy_edge, self.conv_edges))
        output_edges = list(map(copy_edge, self.output_edges))
        
        layer_map = {}
        for innovation_number, layer in self.layer_map.items():
            layer_map[innovation_number] = layer.copy()

        # Make sure we're using the same object for input_edge and output_edge even though it probably doesn't matter
        input_layer: InputLayer = cast(InputLayer, layer_map[self.input_layer.layer_innovation_number])
        assert type(input_layer) == InputLayer
        
        output_layer: OutputLayer = cast(OutputLayer, layer_map[self.output_layer.layer_innovation_number])
        assert type(output_layer) == OutputLayer

        return CnnGenome(   self.number_outputs, input_layer, output_layer, layer_map, conv_edges, output_edges,
                            epigenetic_weights=self.epigenetic_weights, fitness=self.fitness)

    def create_model(self):
        input_layer = self.input_layer.get_tf_layer(self.layer_map, self.edge_map)
        output_layer = self.output_layer.get_tf_layer(self.layer_map, self.edge_map)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    

    def train(self):
        logging.debug("called unimplemented method 'CnnGenome::train'")

        logging.info(f"beginning training of model with initial fitness of {self.fitness}")
        
        dataset = get_dataset()

        # Construct tensorflow model
        model: keras.Model = self.create_model()
        
        # set any epigenetic weights
        if self.epigenetic_weights:
            logging.info("inheriting epigenetic weights")
        for layer_name, weights in self.epigenetic_weights.items():
            model.get_layer(layer_name).set_weights(weights)

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Train it for some set number of epochs
        history = model.fit(dataset.x_train, dataset.y_train, batch_size=128, epochs=2, validation_data=(dataset.x_test, dataset.y_test), verbose=0)

        # Check the fitness
        fitness = history.history['loss'][-1]

        # set the fitness
        self.fitness = fitness

        logging.info(f"finished training model with final fitness of {self.fitness}")

        new_weights = {}

        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                new_weights[layer.name] = layer.get_weights()

        self.epigenetic_weights = new_weights
