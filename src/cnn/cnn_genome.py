import logging
from typing import List, Dict, Any, Optional, Tuple, Set, cast

from tensorflow import keras
import numpy as np

from cnn.cnn_util import make_edge_map
from cnn import Edge
from cnn import ConvEdge
from cnn import DenseEdge
from cnn import Layer
from cnn import InputLayer
from cnn import OutputLayer

import hp


class CnnGenome:


    def __init__(self,  number_outputs: int, input_layer: InputLayer, output_layer: OutputLayer,
                        layer_map: Dict[int, Layer], conv_edges: List[ConvEdge], output_edges: List[DenseEdge],
                        epigenetic_weights: Dict[str, Any]={}, fitness: float=float('inf'), history: Optional[Any]=None,
                        accuracy: float=-0.0):
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
        self.accuracy: float = accuracy 
        self.history: Optional[Any] = history.copy() if history else None

        # Not sure what type the weights will be
        self.epigenetic_weights: Dict[str, Any] = epigenetic_weights
  
        self.island: int = -1


    def copy(self) -> 'CnnGenome':        
        layer_map = {}
        for innovation_number, layer in self.layer_map.items():
            copy = layer.copy()
            
            copy.inputs = set()
            copy.outputs = set()

            layer_map[innovation_number] = copy

        copy_edge = lambda edge: edge.copy(layer_map)
        
        conv_edges = list(map(copy_edge, self.conv_edges))
        output_edges = list(map(copy_edge, self.output_edges))

        # Make sure we're using the same object for input_edge and output_edge even though it probably doesn't matter
        input_layer: InputLayer = cast(InputLayer, layer_map[self.input_layer.layer_innovation_number])
        assert type(input_layer) == InputLayer
        
        output_layer: OutputLayer = cast(OutputLayer, layer_map[self.output_layer.layer_innovation_number])
        assert type(output_layer) == OutputLayer

        return CnnGenome(   self.number_outputs, input_layer, output_layer, layer_map, conv_edges, output_edges,
                            epigenetic_weights=self.epigenetic_weights, fitness=self.fitness, accuracy=self.accuracy, 
                            history=self.history)

        
    def path_exists(self, src: Layer, dst: Layer) -> bool:
        """
        returns True if there is a path from src to dst, otherwise false
        """

        # Set of layer innovation numbers that have been visited.
        visited: Set[int] = set([src.layer_innovation_number])

        # Edges we should traverse
        edges_to_visit: List[int] = list(src.outputs)
    
        # While we've not reached the destination and 
        while dst.layer_innovation_number not in visited and edges_to_visit:
            next_edge_in = edges_to_visit.pop()
            next_edge: Edge = self.edge_map[next_edge_in]
            layer: Layer = self.layer_map[next_edge.output_layer_in]

            if layer.layer_innovation_number in visited:
                continue
            
            if layer.layer_innovation_number == dst.layer_innovation_number:
                return True

            for edge_in in layer.outputs:
                edges_to_visit.append(edge_in)
           
            visited.add(next_edge.output_layer_in)

        return False


    def try_make_new_edge(self, input_layer: Layer, output_layer: Layer) -> Optional[Edge]:
        """
        Attempts to make a new edge but will return None if creating the new edge would lead
        to an invalid neural network graph (i.e. there is a cycle).
        """
        assert input_layer.layer_innovation_number != output_layer.layer_innovation_number
        
        if type(input_layer) == OutputLayer:
            return None

        if self.path_exists(output_layer, input_layer):
            return None

        # if output_layer is the final output layer then we need to make a dense edge
        if type(output_layer) == OutputLayer:
            logging.info(f"creating edge from layer {input_layer.layer_innovation_number} to output layer " + \
                         f"{output_layer.layer_innovation_number}")
            output_edge = DenseEdge(   Edge.get_next_edge_innovation_number(), input_layer.layer_innovation_number, 
                                output_layer.layer_innovation_number, self.layer_map)
            self.output_edges.append(output_edge)
            edge = cast(Edge, output_edge)
        else:
            logging.info(f"creating edge from layer {input_layer.layer_innovation_number} to layer " + \
                         f"{output_layer.layer_innovation_number}")
            conv_edge = ConvEdge(  Edge.get_next_edge_innovation_number(), 1, input_layer.layer_innovation_number,
                                    output_layer.layer_innovation_number, self.layer_map)
            self.conv_edges.append(conv_edge)
            edge = cast(Edge, conv_edge)
        
        self.edge_map[edge.edge_innovation_number] = cast(Edge, edge)

        return edge
    

    def get_two_random_layers(self, rng: np.random.Generator) -> Tuple[Layer, Layer]:
        layers: List[int] = list(self.layer_map.keys())
        
        layer_1_index: int = rng.integers(0, len(layers))
        
        layer_2_index: int = rng.integers(0, len(layers) - 1)
        if layer_2_index >= layer_1_index:
            layer_2_index += 1

        return self.layer_map[layers[layer_1_index]], self.layer_map[layers[layer_2_index]]


    def add_edge_mut(self, rng: np.random.Generator):
        """
        This performs an add edge mutation by randomly selecting two layers and trying to create an edge
        between them. If an edge cannot be created, two different layers will be selected. 
        This process will be repeated until an edge is successfully created.
        """

        # keep trying until we successfully create a new edge
        while True:
            input_layer, output_layer = self.get_two_random_layers(rng)
            edge: Optional[Edge] = self.try_make_new_edge(input_layer, output_layer)

            if edge:
                break


    def try_make_new_layer(self, upper_bound_layer: Layer, lower_bound_layer: Layer, rng: np.random.Generator) -> Optional[Layer]:
        """
        When making a new layer, you must specify the volume size. Thus, two other layers are required to
        randomly select a valid volume size - a minimum and maximum volume size. If the lower bound layer is
        an output layer then only the upper bound layer will be considered.
        """
        if type(upper_bound_layer) == OutputLayer:
            return None

        depth: int = hp.get_random_volume_depth(rng)
        
        upper_width, upper_height, upper_depth = upper_bound_layer.output_shape
        assert upper_width == upper_height
        
        # assume a square volume size
        if type(lower_bound_layer) == OutputLayer:
            # Minimum square size must be at least 2, if the upper bound is two we cant create a smaller layer.
            if upper_width <= 2:
                return None

            volume_size = rng.integers(2, upper_width)
            width, height = volume_size, volume_size
        else:
            lower_width, lower_height, lower_depth = lower_bound_layer.output_shape
            assert lower_width == lower_height
            
            if upper_width < lower_width:
                return None

            # If this is 3 then the lower must be 2 and we cannot create a size inbetween 2 and 3
            if upper_width <= 3:
                assert lower_width == 2
                return None
            logging.info(f"lower = {lower_width}, upper = {upper_width}")
            volume_size = rng.integers(lower_width, upper_width)
            width, height = volume_size, volume_size
        
        logging.info(f"volume size is {width}, {height}, {depth}")

        layer = Layer(Layer.get_next_layer_innovation_number(), width, height, depth)
        
        assert layer.layer_innovation_number not in self.layer_map
        self.layer_map[layer.layer_innovation_number] = layer
        
        return layer

    
    def add_layer_mut(self, rng: np.random.Generator):
        """
        This performs an add layer mutation by selecting two random layers and creating a layer that connects
        the two selected layers with two additional edges. If connecting the selected input layer to the
        output layer would create a cycle then two different layers are selected. 
        This will be repeated until two valid layers are selected
        """
   
        # keep trying until we successfully find two layers we can connect
        while True:
            input_layer, output_layer = self.get_two_random_layers(rng)
            
            # If this is the case adding an edge would create a cycle
            if self.path_exists(output_layer, input_layer):
                continue

            input_width, input_height, input_depth = input_layer.output_shape
            output_width, output_height, output_depth = output_layer.output_shape
            
            # This could lead to a negative filter size / invalid
            if input_width < output_width or input_height < output_height:
                continue

            maybe_layer: Optional[Layer] = self.try_make_new_layer(input_layer, output_layer, rng)

            if maybe_layer:
                break
        
        layer: Layer = cast(Layer, maybe_layer)

        # Assertions here because these should not fail
        assert self.try_make_new_edge(input_layer, layer)
        assert self.try_make_new_edge(layer, output_layer)


    def create_model(self):
        input_layer = self.input_layer.get_tf_layer(self.layer_map, self.edge_map)
        output_layer = self.output_layer.get_tf_layer(self.layer_map, self.edge_map)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    

    def train(self):
        logging.debug("called unimplemented method 'CnnGenome::train'")
        
        if self.history:
            logging.info(f"beginning training of model with initial accuracy of {self.history['categorical_accuracy'][-1]:.6f}, fitness = {self.fitness:.6f}")
        else:
            logging.info(f"beginning training of new model")

        dataset = hp.get_dataset()

        # Construct tensorflow model
        model: keras.Model = self.create_model()
        logging.info(f"model has {model.count_params()} parameters")
        
        # set any epigenetic weights
        if self.epigenetic_weights:
            logging.info("inheriting epigenetic weights")
        for layer_name, weights in self.epigenetic_weights.items():
            model.get_layer(layer_name).set_weights(weights)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Train it for some set number of epochs
        history = model.fit(dataset.x_train, dataset.y_train, batch_size=512, epochs=1, validation_data=(dataset.x_test, dataset.y_test), verbose=0)

        # Check the fitness
        fitness = history.history['loss'][-1]
        accuracy = history.history['categorical_accuracy'][-1]

        # set the fitness
        self.fitness = fitness
        self.accuracy = accuracy
        self.history = history.history

        logging.info(f"finished training of model with final accuracy of {accuracy:.6f}, fitness = {self.fitness:.6f}")

        new_weights = {}

        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                new_weights[layer.name] = layer.get_weights()

        self.epigenetic_weights = new_weights
