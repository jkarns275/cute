import logging

from typing import List

class CnnGenome:
    
    def __init__(self):
        #  , conv_layers: List[CnnLayer], conv_edges: List[ConvEdge],
        #           fc_layers: List[FullyConnectedLayer], edges: List[Edge]):
        # TODO: Ask about how we should parmaeterize this and how we should name it.
        #       1. Operations should be stored in the conv edges (but they're layer based edges, meaning
        #           they connect entire layers with entire other layers)
        #       2. Fully connected layers should maybe be separate from the convolutional layers?
        #           And then these edges are also layer based
        #       3. How do we inhereit the weights from fully connected layers properly?
        #           Need to figure out how tensorflow stores the weights / in what order and how adding
        #           an additional input layer would change that.
        pass


    def train(self):
        logging.debug("called unimplemented method 'CnnGenome::train'")
        # raise NotImplementedError
