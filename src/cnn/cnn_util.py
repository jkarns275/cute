from typing import Tuple, Optional, List
import logging

if False:
    from cnn import Edge, Layer


def calculate_required_filter_size(
        stride: int,
        input_width: int, input_height: int, input_depth: int,
        output_width: int, output_height: int, output_depth: int) -> Tuple[int, int]:
    # if the two layers can be connected, this will return a tuple of the required filter size
    # as a tuple (filter_width, filter_height). If the two layers cannot be connected then 
    # this will return None
    f_w = input_width + stride * (1 - output_width)
    f_h = input_height + stride * (1 - output_height)

    return f_w, f_h


def calculate_output_volume_size(stride: int, padding: int, filter_width: int, filter_height: int, 
                                 number_filters: int, width: int, height: int, depth: int) -> Tuple[int, int, int]:
    """
    Calculates the output volume size for a convolution operation with the specified stride, filter size, number of
    filters, and input volume width, height, and depth.
    """
    w = (width - filter_width + 2 * padding) // stride + 1
    h = (height - filter_height + 2 * padding) // stride + 1
    
    return w, h, number_filters


def make_layer_map(layers: List['Layer']):
    layer_map = {}

    for layer in layers:
        assert layer.layer_innovation_number not in layer_map
        layer_map[layer.layer_innovation_number] = layer

    return layer_map


def make_edge_map(edges: List['Edge']):
    edge_map = {}

    for edge in edges:
        assert edge.edge_innovation_number not in edge_map
        edge_map[edge.edge_innovation_number] = edge

    return edge_map


def get_possible_strides(   input_width: int, input_height: int, _input_depth: int,
                            output_width: int, output_height: int, _output_depth: int) -> List[int]:
    assert input_width == input_height
    # I think this is the correct calculation...
    max_stride = max(-input_width // (1 - output_width) - 1, 1)

    return list(range(1, max_stride + 1))
