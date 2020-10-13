import logging
from typing import List, Tuple, Optional, Dict, Set

import tensorflow.keras as keras
import tensorflow as tf

from hp import make_activation_layer, make_classification_layer, get_regularizer
from cnn.edge import Edge
from cnn.layer import Layer
if False:
    from cnn import CnnGenome

class OutputLayer(Layer):
    """
    This is a subclass of Layer so that the add_input, __getstate__, and __setstate__ methods are inherited.
    Also inherited is the 'inputs' field, which is a list of the input layers
    """


    def __init__(self, layer_innovation_number: int, dense_layers: List[int], number_classes: int, 
                       inputs: Set[int]=set(), outputs: Set[int]=set()): 
        """
        Parameters
        ----------
        dense_layers    : :obj:`list` of :obj:`int`
            A list of fully connected layer sizes which will be placed at the end of the network. 
            If this list is empty the only fully connected layer will be the softmax layer.
        number_classes  : int
            The number of possible output classes. This means the final layer in the network
            will have `number_classes` nodes.
        """
        super().__init__(layer_innovation_number, number_classes, 1, 1, inputs=inputs, outputs=outputs)
       
        self.layer_innovation_number: int = layer_innovation_number
        self.number_classes: int = number_classes
        self.dense_layers: List[int] = dense_layers + [number_classes]
        self.tf_weight_names: Set[str] = set()

    
    def copy(self) -> 'OutputLayer':
        return OutputLayer( self.layer_innovation_number, self.dense_layers[:-1], self.number_classes, 
                            inputs=self.inputs, outputs=self.outputs)


    def get_tf_weight_names(self) -> Set[str]:
        weight_names = set()
        name = self.get_name()

        for i, size in enumerate(self.dense_layers):
            weight_names.add(f"{name}_{i}")
        
        return weight_names

    def set_enabled(self, enabled: bool):
        self.enabled = True


    def get_enabled(self):
        return True


    def get_first_layer_size(self):
        return self.dense_layers[0]

    
    def get_name(self):
        return f"output_layer_inov_n_{self.layer_innovation_number}"


    def get_tf_layer(self, genome: 'CnnGenome') -> keras.layers.Layer:
        """
        A description of how this method works to construct a complete TensorFlow computation graph can be found
        in the documentation for the CnnGenome::create_model.
        
        This should never return None. Takes all of the tensors from DenseEdges and adds them together, then pushes
        them through an activation function. Then adds a series of fully connected layers then the classification layer.
        Returns this as a Tensor of course.
        """


        if self.tf_layer is not None:
            return self.tf_layer
        
        # To make it easy to inherit epigenetic weights, we seperate the weights out into separate
        # layers without activation functions, and add the resulting layers together and then apply
        # an activation function.
        # So these intermediate layers shouldn't have an activation function
        maybe_intermediate_layers: List[Optional[tf.Tensor]] = list(map(lambda edge_in: genome.edge_map[edge_in].get_tf_layer(genome), self.inputs))
        
        # filter out nones
        intermediate_layers: List[tf.Tensor] = [x for x in maybe_intermediate_layers if x is not None]
        
        if not intermediate_layers:
            logging.fatal(f"output layer has no inputs!")

        assert intermediate_layers

        layer: tf.Tensor = None

        # All but the last dense layer should have the hyper parameter specified activation type.
        # The last layer should be a classification activation type life softmax or svm
        for i, size in enumerate(self.dense_layers):
            
            # if layer is none we haven't used the intermediate layers yet, so we have to sum them if
            # there is more than one (if theres only one it will throw an error)
            if layer is None:
                if len(intermediate_layers) > 1:
                    layer = keras.layers.Add()(intermediate_layers)
                else:
                    layer = intermediate_layers[0]           
            else:
                shape = layer.shape[1:]
                layer = \
                    keras.layers.Dense(
                            size,
                            input_shape=shape,
                            activation='linear',
                            kernel_regularizer=get_regularizer(genome.hp),
                            bias_regularizer=get_regularizer(genome.hp),
                            name=self.get_name() + f"_{i}")(layer)

            if i == len(self.dense_layers) - 1:
                layer = make_classification_layer()(layer)
            else:
                layer = make_activation_layer()(layer)

        self.tf_layer = layer

        return self.tf_layer
