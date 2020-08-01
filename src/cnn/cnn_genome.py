import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Iterator, cast, Callable
from itertools import product, chain

from tensorflow import keras
import numpy as np

from cnn.cnn_util import make_edge_map, get_possible_strides
from cnn import Edge, ConvEdge, SeparableConvEdge, FractionalMaxPoolingEdge, DenseEdge
from cnn import Layer, InputLayer, OutputLayer

import hp


class CnnGenome:


    EDGE_TYPE_FUNCTIONS: List[Callable] = []
    EDGE_TYPE_PROBABILITIES: List[float] = []


    @staticmethod
    def init():
        # Maps a mutate function to its probability so the rng can select one randomly
        edge_type_probability_map = {
            CnnGenome.add_conv_edge_mut:             hp.add_conv_edge_probability,
            CnnGenome.add_separable_conv_edge_mut:   hp.add_separable_conv_edge_probability,
            CnnGenome.add_pooling_edge_mut:          hp.add_pooling_edge_probability,
            CnnGenome.add_dense_edge_mut:            hp.add_dense_edge_probability
        }
        
        for mutation_function, mutation_probability in edge_type_probability_map.items():
            CnnGenome.EDGE_TYPE_FUNCTIONS.append(mutation_function)
            CnnGenome.EDGE_TYPE_PROBABILITIES.append(mutation_probability)


    @staticmethod
    def try_crossover(rng: np.random.Generator, *parents_tuple: 'CnnGenome') -> Optional['CnnGenome']:
        """
        Attempts to perform crossover. This could fail in the event that the resulting child genome has no path
        from the input layer to the output layer. This won't happen if the parent genome's accept_rate is 1.0
        """
        parents: List['CnnGenome'] = list(parents_tuple)
        assert len(parents) > 1
        
        def sort_key(genome: 'CnnGenome') -> float:
            return genome.fitness

        parents = list(sorted(parents, key=sort_key))

        logging.info(f"attempting crossover with {len(parents)} parents")
        for i, parent in enumerate(parents):
            logging.info(   f"parent {i} has {parent.number_enabled_edges()} enabled edges and " + \
                            f"{parent.number_enabled_layers()} enabled layers")

        layer_map: Dict[int, Layer] = {}
        edge_map: Dict[int, Edge] = {}
        epigenetic_weights: Dict[str, Any] = {}
        disabled_edges: Set[int] = set()
        disabled_layers: Set[int] = set()


        def try_add_layer(layer: Layer, enabled: bool=True):
            """
            Add a copy of the supplied layer to the layer map of the new genome we are constructing,
            enabling or disabling it based on the value of enabled
            """
            if layer.layer_innovation_number not in layer_map:
                copy = layer.copy()
                copy.clear_io()
                copy.set_enabled(enabled)
                layer_map[layer.layer_innovation_number] = copy
                if not enabled:
                    disabled_layers.add(layer.layer_innovation_number)
            elif enabled and layer.layer_innovation_number in disabled_layers:
                layer_map[layer.layer_innovation_number].set_enabled(enabled)
                disabled_layers.remove(layer.layer_innovation_number)

        def try_add_edge(edge: Edge, enabled: bool=True):
            """
            Add a copy of the supplied edge to the edge map, enabling or disabling it based on the value
            of enabled
            """
            if  edge.input_layer_in in layer_map and \
                edge.output_layer_in in layer_map:
                
                if edge.edge_innovation_number not in edge_map:
                    copy = edge.copy(layer_map)
                    copy.set_enabled(enabled)
                    edge_map[edge.edge_innovation_number] = copy
                    if not enabled:
                        disabled_edges.add(edge.edge_innovation_number)

                elif enabled and edge.edge_innovation_number in disabled_edges:
                    edge_map[edge.edge_innovation_number].set_enabled(enabled)
                    disabled_edges.remove(edge.edge_innovation_number)

        for i, parent in enumerate(parents):
            accept_rate = hp.get_crossover_accept_rate(i)

            for layer in parent.layer_map.values():
                sample = rng.random()
                try_add_layer(layer, sample <= accept_rate and layer.is_enabled())

            for edge in parent.edge_map.values():
                sample = rng.random()
                try_add_edge(edge, sample <= accept_rate and layer.is_enabled())

            # We will get the "best" weights here because we sorted by genome fitness - the first things added
            # will have the best fitness
            for name, weights in parent.epigenetic_weights.items():
                if name not in epigenetic_weights:
                    epigenetic_weights[name] = weights

        conv_edges: List[ConvEdge] = []
        output_edges: List[DenseEdge] = []
        for edge in edge_map.values():
            ty = type(edge)
            if issubclass(ty, ConvEdge):
                conv_edges.append(cast(ConvEdge, edge))
            elif ty == DenseEdge:
                output_edges.append(cast(DenseEdge, edge))
            else:
                raise RuntimeError(f"unrecognized edge type '{ty}'")

        number_outputs: int = parents[0].number_outputs
        input_layer: InputLayer = cast(InputLayer, layer_map[parents[0].input_layer.layer_innovation_number])
        output_layer: OutputLayer = cast(OutputLayer, layer_map[parents[0].output_layer.layer_innovation_number])
         
        # For now use default fitness, history, and accuracy
        # Maybe we'll want to use the ones from the parent genome
        child = CnnGenome(  number_outputs, input_layer, output_layer, layer_map, conv_edges, output_edges,
                            epigenetic_weights, disabled_layers, disabled_edges)

        if child.path_exists(child.input_layer, child.output_layer, False):
            logging.info("crossover succeeded!")
            logging.info(  f"child has {child.number_enabled_edges()} enabled edges and " + \
                            f"{child.number_enabled_layers()} enabled layers")
            return child
        else:
            logging.info("crossover failed because there was no path from the input layer to the output layer")
            return None


    def __init__(self,  number_outputs: int, input_layer: InputLayer, output_layer: OutputLayer,
                        layer_map: Dict[int, Layer], conv_edges: List[ConvEdge], output_edges: List[DenseEdge],
                        epigenetic_weights: Dict[str, Any], disabled_layers: Set[int], disabled_edges: Set[int], 
                        fitness: float=float('inf'), history: Optional[Any]=None, accuracy: float=-0.0):
        # When this object is serialized the input and output layer in this map are copied so we need to make sure
        # we use the same object, otherwise layer_map[input_layer.layer_innovation_number] and input_layer will be
        # equal but different objects.
        layer_map[input_layer.layer_innovation_number] = input_layer
        layer_map[output_layer.layer_innovation_number] = output_layer

        self.conv_edges: List[ConvEdge] = conv_edges
        for conv_edge in conv_edges:
            assert issubclass(type(conv_edge), ConvEdge)

        self.output_edges: List[DenseEdge] = output_edges
        for output_edge in output_edges:
            assert type(output_edge) == DenseEdge

        self.edge_map: Dict[int, Edge] = make_edge_map(cast(List[Edge], self.conv_edges) + cast(List[Edge], self.output_edges))

        self.number_outputs: int = number_outputs
        self.fully_connected_layers: List[int] = [1024, number_outputs]
        
        self.input_layer: InputLayer = input_layer
        self.output_layer: OutputLayer = output_layer
        self.layer_map: Dict[int, Layer] = layer_map
        self.disabled_layers: Set[int] = disabled_layers
        self.disabled_edges: Set[int] = disabled_edges

        self.fitness: float = fitness
        self.accuracy: float = accuracy 
        self.history: Optional[Any] = history

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
        
        # TODO: might be able to remove some of these copies
        return CnnGenome(   self.number_outputs, input_layer, output_layer, layer_map, conv_edges.copy(), output_edges.copy(),
                            epigenetic_weights=self.epigenetic_weights.copy(), disabled_layers=self.disabled_layers.copy(), 
                            disabled_edges=self.disabled_edges.copy(), fitness=self.fitness, accuracy=self.accuracy, history=self.history)


    def __eq__(self, o: object) -> bool:
        if type(o) == CnnGenome:
            if self.disabled_edges != o.disabled_edges:
                return False
            elif self.disabled_layers != o.disabled_layers:
                return False
            elif set(self.layer_map.keys()) != set(o.layer_map.keys()):
                return False
            elif set(self.edge_map.keys()) != set(o.edge_map.keys()):
                return False

            return True
        else:
            return False


    def path_exists(self, src: Layer, dst: Layer, include_disabled=True) -> bool:
        """
        returns True if there is a path from src to dst, otherwise false
        """
        
        if src.layer_innovation_number == dst.layer_innovation_number:
            return True

        # Set of layer innovation numbers that have been visited.
        visited: Set[int] = {src.layer_innovation_number}

        # Edges we should traverse
        edges_to_visit: List[int] = list(src.outputs)
        
        # While we've not reached the destination and 
        while dst.layer_innovation_number not in visited and edges_to_visit:
            next_edge_in = edges_to_visit.pop()
            next_edge: Edge = self.edge_map[next_edge_in]
            
            if not include_disabled and next_edge_in in self.disabled_edges:
                continue
            
            if next_edge.output_layer_in in visited:
                continue
            
            layer: Layer = self.layer_map[next_edge.output_layer_in]
            
            if not layer.is_enabled():
                continue

            if layer.layer_innovation_number == dst.layer_innovation_number:
                return True

            for edge_in in layer.outputs:
                if include_disabled or edge_in not in self.disabled_edges:
                    edges_to_visit.append(edge_in)
           
            visited.add(next_edge.output_layer_in)

        return False

    
    def valid_connection(self, input_layer: Layer, output_layer: Layer, ty=None):
        """
        Ty should be a edge type (DenseEdge, ConvEdge, etc), and if it is not None
        we will check for duplicate edges of the same type
        """

        # Output layer cannot be an input layer, it is the final layer
        if type(input_layer) == OutputLayer:
            return False
        
        if type(output_layer) == OutputLayer and ty != DenseEdge:
            return None

        if input_layer.layer_innovation_number == output_layer.layer_innovation_number:
            return False

        # No cycles
        if self.path_exists(output_layer, input_layer):
            return False
    
        if ty is not None:
            for edge_in, edge in self.edge_map.items():
                if  edge.input_layer_in == input_layer.layer_innovation_number and \
                    edge.output_layer_in == output_layer.layer_innovation_number and \
                    type(edge) == ty:
                    return False

        return True


    def register_edge(self, edge: Edge):
        if issubclass(type(edge), ConvEdge):
            for conv_edge in self.conv_edges:
                assert conv_edge.edge_innovation_number != edge.edge_innovation_number

            self.conv_edges.append(cast(ConvEdge, edge))
        elif issubclass(type(edge), DenseEdge):
            for dense_edge in self.output_edges:
                assert dense_edge.edge_innovation_number != edge.edge_innovation_number

            self.output_edges.append(cast(DenseEdge, edge))
        
        assert edge.edge_innovation_number not in self.edge_map
        self.edge_map[edge.edge_innovation_number] = edge

    
    def try_make_new_conv_edge(self, input_layer: Layer, output_layer: Layer, rng: np.random.Generator, conv_edge_type=ConvEdge) -> Optional[Edge]:
        if not self.valid_connection(input_layer, output_layer):
            return None

        # No negative filter sizes
        input_width, input_height, input_depth = input_layer.output_shape
        output_width, output_height, output_depth = output_layer.output_shape
       
        if input_width < output_width or input_height < output_height:
            return None
        
        possible_strides: List[int] = get_possible_strides(*input_layer.output_shape, *output_layer.output_shape)
        
        # No duplicate output edges with the same stride
        for edge_in in input_layer.outputs:
            edge = self.edge_map[edge_in]
            if  edge.input_layer_in == input_layer.layer_innovation_number and \
                edge.output_layer_in == output_layer.layer_innovation_number and \
                type(edge) == conv_edge_type:
                conv_edge: ConvEdge = cast(ConvEdge, edge)
                if conv_edge.stride in possible_strides:
                    possible_strides.remove(conv_edge.stride)

        if not possible_strides:
            return None

        stride: int = possible_strides[rng.integers(0, len(possible_strides))]
        
        conv_edge = conv_edge_type( Edge.get_next_edge_innovation_number(), stride, input_layer.layer_innovation_number,
                                    output_layer.layer_innovation_number, self.layer_map)
        self.register_edge(conv_edge)
        
        edge = cast(Edge, conv_edge)
        
        logging.info(f"creating separable conv edge from layer {input_layer.layer_innovation_number} to layer " + \
                     f"{output_layer.layer_innovation_number}")

        return edge
    

    def try_make_new_pooling_edge(self, input_layer: Layer, output_layer: Layer, rng: np.random.Generator, pooling_edge_type=FractionalMaxPoolingEdge) -> Optional[Edge]:
        if not self.valid_connection(input_layer, output_layer, FractionalMaxPoolingEdge):
            return None
        
        iw, ih, id = input_layer.output_shape
        ow, oh, od = output_layer.output_shape

        # In tensorflow the size must be less than and input channels should be the same as output channels,
        # but if there are moer output channels we will just use padding
        # Okay just kidding this padding thing is tough
        if ow >= iw or oh >= ih or id > od:
            return None

        # No duplicate output edges with the same stride
        for edge_in in input_layer.outputs:
            edge = self.edge_map[edge_in]
            if  edge.input_layer_in == input_layer.layer_innovation_number and \
                edge.output_layer_in == output_layer.layer_innovation_number:
                if type(edge) == FractionalMaxPoolingEdge:
                    return None
        pooling_edge: ConvEdge = pooling_edge_type( Edge.get_next_edge_innovation_number(), input_layer.layer_innovation_number,
                                                    output_layer.layer_innovation_number, self.layer_map)

        self.register_edge(pooling_edge)
        edge = cast(Edge, pooling_edge)

        return edge
        

    def try_make_new_dense_edge(self, input_layer: Layer, rng: np.random.Generator) -> Optional[Edge]:
        """
        Attempts to make a new edge but will return None if creating the new edge would lead
        to an invalid neural network graph (i.e. there is a cycle).
        The stride is randomly selected from all possible strides (unless a DenseEdge is created, which has no stride).
        """
        output_layer = self.output_layer
        if not self.valid_connection(input_layer, output_layer, DenseEdge):
            return None

        logging.info(f"creating densae edge from layer {input_layer.layer_innovation_number} to output layer " + \
                     f"{output_layer.layer_innovation_number}")
        output_edge = DenseEdge(Edge.get_next_edge_innovation_number(), input_layer.layer_innovation_number, 
                                output_layer.layer_innovation_number, self.layer_map)
        self.register_edge(output_edge)
        return cast(Edge, output_edge)
        
    
    def try_make_new_separable_conv_edge(self, input_layer: Layer, output_layer: Layer, rng: np.random.Generator) -> Optional[Edge]:
        edge = self.try_make_new_conv_edge(input_layer, output_layer, rng, conv_edge_type=SeparableConvEdge)

        if not edge:
            return None

        self.edge_map[edge.edge_innovation_number] = cast(Edge, edge)
        
        return edge


    def try_make_new_random_edge(self, input_layer: Layer, output_layer: Layer, rng: np.random.Generator) -> Optional[Edge]:
        if type(output_layer) == OutputLayer:
            return self.try_make_new_dense_edge(input_layer, rng)

        mutation_operations = rng.choice(CnnGenome.EDGE_TYPE_FUNCTIONS, len(CnnGenome.EDGE_TYPE_FUNCTIONS), p=CnnGenome.EDGE_TYPE_PROBABILITIES, replace=False)

        for operation in mutation_operations:
            edge = None
            if operation == CnnGenome.add_conv_edge_mut:
                edge = self.try_make_new_conv_edge(input_layer, output_layer, rng)
            elif operation == CnnGenome.add_separable_conv_edge_mut:
                edge = self.try_make_new_separable_conv_edge(input_layer, output_layer, rng)
            elif operation == CnnGenome.add_pooling_edge_mut:
                edge = self.try_make_new_pooling_edge(input_layer, output_layer, rng)

            if edge:
                return edge
               

        return None


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
            
            if upper_width <= lower_width:
                return None

            # If this is 3 then the lower must be 2 and we cannot create a size inbetween 2 and 3
            if upper_width <= 3:
                assert lower_width == 2
                return None

            volume_size = rng.integers(lower_width, upper_width)
            width, height = volume_size, volume_size
        

        layer = Layer(Layer.get_next_layer_innovation_number(), width, height, depth)
        
        assert layer.layer_innovation_number not in self.layer_map
        self.layer_map[layer.layer_innovation_number] = layer
        
        return layer


    def random_layer_iterator(self, rng: np.random.Generator) -> Iterator[int]:
        layer_ins = list(self.layer_map.keys())

        rng.shuffle(layer_ins)
        
        return cast(Iterator[int], layer_ins)

 
    def random_layer_pair_iterator(self, rng: np.random.Generator) -> Iterator[Tuple[int, int]]:
        return product(self.random_layer_iterator(rng), self.random_layer_iterator(rng))


    def random_edge_iterator(self, rng: np.random.Generator) -> Iterator[int]:
        get_in = lambda edge: edge.edge_innovation_number
        all_edges: List[int] = list(map(get_in, self.conv_edges)) + list(map(get_in, self.output_edges))
        rng.shuffle(all_edges)

        return cast(Iterator[int], all_edges)
 

    def enable_edge(self, edge_in: int):
        if edge_in not in self.disabled_edges:
            logging.info(f"attempted to enable edge {edge_in} that was already enabled")

        self.disabled_edges.remove(edge_in)
        self.edge_map[edge_in].enable()


    def disable_edge(self, edge_in: int):
        if edge_in in self.disabled_edges:
            logging.info(f"attempted to disable {edge_in} that was already disabled")

        self.disabled_edges.add(edge_in)
        self.edge_map[edge_in].disable()


    def enable_layer(self, layer_in: int):
        if layer_in not in self.disabled_layers:
            logging.info(f"attempted to enable {layer_in} that was already enabled")

        self.disabled_layers.remove(layer_in)
        self.layer_map[layer_in].enable()


    def disable_layer(self, layer_in: int):
        if layer_in in self.disabled_layers:
            logging.info(f"attempted to disable {layer_in} that was already disabled")

        self.disabled_layers.add(layer_in)
        self.layer_map[layer_in].disable()


    def add_edge_mut(self, rng: np.random.Generator) -> bool:
        """
        This performs an add edge mutation by randomly selecting two layers and trying to create an edge
        between them. If an edge cannot be created, two different layers will be selected. 
        This process will be repeated until an edge is successfully created.
        """
        logging.info("attempting add_edge mutation")

        mutation_operations = rng.choice(CnnGenome.EDGE_TYPE_FUNCTIONS, len(CnnGenome.EDGE_TYPE_FUNCTIONS), p=CnnGenome.EDGE_TYPE_PROBABILITIES, replace=False)

        for operation in mutation_operations:
            if operation(self, rng):
                logging.info("successfully completed add_edge mutation")
                return True
                
        logging.info("failed to complete add_edge mutation")
        return False
    

    def add_dense_edge_mut(self, rng: np.random.Generator) -> bool:
        logging.info("attempting add_dense_edge mutation")
        
        for layer_in in self.random_layer_iterator(rng):
            if self.try_make_new_dense_edge(self.layer_map[layer_in], rng):
                logging.info("successfully completed add_dense_edge mutation")
                return True

        logging.info("failed to complete add_dense_edge mutation")
        return False


    def add_separable_conv_edge_mut(self, rng: np.random.Generator) -> bool:
        """
        This performs an add seperable edge mutation by randomly selecting two layers and trying to create an edge
        between them. If an edge cannot be created, two different layers will be selected. 
        This process will be repeated until an edge is successfully created.

        A separable convolution is one that uses two convolve operations using a nx1 filter and 1xn filter
        to achieve the same output volume size as an nxn filter. It requires 2n parameters as compared to n*n
        """
        logging.info("attempting add_separable_conv_edge mutation")

        # Try every combination until we find one that works, or we exhaust all combinations.
        for input_layer_in, output_layer_in in self.random_layer_pair_iterator(rng):
            input_layer = self.layer_map[input_layer_in]
            output_layer = self.layer_map[output_layer_in]
            
            edge: Optional[Edge] = self.try_make_new_separable_conv_edge(input_layer, output_layer, rng)

            if edge:
                logging.info("successfully completed add_separable_conv_edge mutation")
                return True
        
        logging.info("failed to complete add_separable_conv_edge mutation")
        return False


    def add_conv_edge_mut(self, rng: np.random.Generator) -> bool:
        """
        This performs an add seperable edge mutation by randomly selecting two layers and trying to create an edge
        between them. If an edge cannot be created, two different layers will be selected. 
        This process will be repeated until an edge is successfully created.

        A separable convolution is one that uses two convolve operations using a nx1 filter and 1xn filter
        to achieve the same output volume size as an nxn filter. It requires 2n parameters as compared to n*n
        """
        logging.info("attempting add_separable_conv_edge mutation")

        # Try every combination until we find one that works, or we exhaust all combinations.
        for input_layer_in, output_layer_in in self.random_layer_pair_iterator(rng):
            input_layer = self.layer_map[input_layer_in]
            output_layer = self.layer_map[output_layer_in]
            
            edge: Optional[Edge] = self.try_make_new_conv_edge(input_layer, output_layer, rng)

            if edge:
                logging.info("successfully completed add_separable_conv_edge mutation")
                return True
        
        logging.info("failed to complete add_separable_conv_edge mutation")
        return False


    def add_pooling_edge_mut(self, rng: np.random.Generator) -> bool:
        """
        This attempts to add a pooling edge between two random layers.
        Every combination of layers will be tried, exhaustively, until we successfully connect to layers.
        This process will be repeated until an edge is successfully created.
        """
        logging.info("attempting add_edge mutation")

        # Try every combination until we find one that works, or we exhaust all combinations.
        for input_layer_in, output_layer_in in self.random_layer_pair_iterator(rng):
            input_layer = self.layer_map[input_layer_in]
            output_layer = self.layer_map[output_layer_in]
            
            edge: Optional[Edge] = self.try_make_new_pooling_edge(input_layer, output_layer, rng)

            if edge:
                logging.info("successfully completed add_pooling_edge mutation")
                return True
        
        logging.info("failed to complete add_pooling_edge mutation")
        return False
   

    def add_layer_mut(self, rng: np.random.Generator):
        """
        This performs an add layer mutation by selecting two random layers and creating a layer that connects
        the two selected layers with two additional edges. If connecting the selected input layer to the
        output layer would create a cycle then two different layers are selected. 
        This will be repeated until two valid layers are selected
        """
        logging.info("attempting add_layer mutation")
        
        for input_layer_in, output_layer_in in self.random_layer_pair_iterator(rng):
            input_layer = self.layer_map[input_layer_in]
            output_layer = self.layer_map[output_layer_in]
            
            # If this is the case adding an edge would create a cycle
            if self.path_exists(output_layer, input_layer):
                continue

            input_width, input_height, input_depth = input_layer.output_shape
            output_width, output_height, output_depth = output_layer.output_shape
           
            # This could lead to a negative filter size / invalid
            if input_width < output_width + 4 or input_height < output_height + 4:
                continue

            maybe_layer: Optional[Layer] = self.try_make_new_layer(input_layer, output_layer, rng)

            if maybe_layer:
                layer: Layer = cast(Layer, maybe_layer)

                # Assertions here because these should not fail
                assert self.try_make_new_random_edge(input_layer, layer, rng)

                assert self.try_make_new_random_edge(layer, output_layer, rng)
                
                logging.info("successfully completed add_layer mutation")
                return True
        
        logging.info("failed to complete add_layer mutation")
        return False
    
    
    def enable_edge_mut(self, rng: np.random.Generator) -> bool:
        logging.info("attempting enable_edge mutation")
        
        if not self.disabled_edges:
            logging.info("failed to complete enable_edge mutation")
            return False

        disabled_edges: List[int] = list(self.disabled_edges)
        index: int = rng.integers(0, len(disabled_edges))
        edge_in: int = disabled_edges[index]

        self.enable_edge(edge_in)

        logging.info("successfully completed enable_edge mutation")
        return True

       
    def disable_edge_mut(self, rng: np.random.Generator) -> bool:
        logging.info("attempting disable_edge mutation")
        
        # We will attempt to disable edges until we succeed or we tried every edge.
        for edge_in in self.random_edge_iterator(rng):
            if edge_in in self.disabled_edges:
                continue

            self.disable_edge(edge_in)

            if self.path_exists(self.input_layer, self.output_layer, False):
                edge = self.edge_map[edge_in]
                logging.info(f"disabling edge from {edge.input_layer_in} to {edge.output_layer_in}")
                logging.info("successfully completed disable_edge mutation")
                return True

            self.enable_edge(edge_in)
        
        logging.info("failed to complete disable_edge mutation")
        return False
   

    def disable_layer_mut(self, rng: np.random.Generator) -> bool:
        logging.info("attempting disable_layer mutation")
        
        layers_to_ignore = {self.input_layer.layer_innovation_number, self.output_layer.layer_innovation_number}
        for layer_in in self.random_layer_iterator(rng):
            if layer_in in self.disabled_layers or layer_in in layers_to_ignore:
                continue

            self.disable_layer(layer_in)

            if self.path_exists(self.input_layer, self.output_layer, False):
                logging.info(f"disabling layer {layer_in}")
                logging.info("successfully completed disable_layer mutation")
                return True

            self.enable_layer(layer_in)
        
        logging.info("failed to complete disable_layer mutation")
        return False
    

    def enable_layer_mut(self, rng: np.random.Generator) -> bool:
        logging.info("attempting enable_layer mutation")

        if not self.disabled_layers:
            logging.info("failed to complete enable_layer mutation")
            return False

        disabled_layers = list(self.disabled_layers)
        index: int = rng.integers(0, len(disabled_layers))
        layer_in: int = disabled_layers[index]

        self.enable_layer(layer_in)
        
        logging.info(f"enabling layer {layer_in}")
        logging.info("successfully completed enable_layer mutation")
        return True


    def copy_mut(self, _rng: np.random.Generator):
        logging.info("successfully completed copy mutation")
        return True


    def number_enabled_layers(self):
        n = 0
        for i, layer in self.layer_map.items():
            if layer.is_enabled():
                n += 1
        return n

    
    def number_enabled_edges(self):
        n = 0
        for i, layer in self.edge_map.items():
            if layer.is_enabled():
                n += 1
        return n


    def create_model(self):
        input_layer = self.input_layer.get_tf_layer(self.layer_map, self.edge_map)
        output_layer = self.output_layer.get_tf_layer(self.layer_map, self.edge_map)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    

    def train(self):
        if self.history:
            logging.info(f"beginning training of model with initial accuracy of {self.accuracy:.6f}, fitness = {self.fitness:.6f}")
        else:
            logging.info(f"beginning training of new model")

        dataset = hp.get_dataset()

        # Construct tensorflow model
        model: keras.Model = self.create_model()
        logging.info(f"model has {model.count_params()} parameters")
        model.summary()

        # set any epigenetic weights
        if self.epigenetic_weights:
            logging.info("inheriting epigenetic weights")

        for layer_name, weights in self.epigenetic_weights.items():
            try:
                model.get_layer(layer_name).set_weights(weights)
            except ValueError as _ve:
                # The layer was disabled
                pass

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
       
        # loss, acc = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        # logging.info(f"calculated acc {acc}")

        # Train it for some set number of epochs
        history = model.fit(dataset.x_train, dataset.y_train, batch_size=hp.get_batch_size(), epochs=hp.get_number_epochs(), validation_data=(dataset.x_test, dataset.y_test), verbose=0)

        # Check the fitness
        fitness = 1.0 / history.history['val_categorical_accuracy'][-1]
        accuracy = history.history['val_categorical_accuracy'][-1]

        # set the fitness
        self.fitness = fitness + hp.PARAMETER_COUNT_PENALTY_WEIGHT * model.count_params()
        self.accuracy = accuracy
        self.history = history.history

        logging.info(f"finished training of model with final accuracy of {accuracy:.6f}, fitness = {self.fitness:.6f}")

        new_weights = {}

        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                new_weights[layer.name] = layer.get_weights()

        self.epigenetic_weights.update(new_weights)


CnnGenome.init()
