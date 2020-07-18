import os
import sys
import pickle
import logging
from typing import List
# This hides tensorflow debug output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

from mpi4py import MPI
import tensorflow as tf

import hp
from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer, SeparableConvEdge, FractionalMaxPoolingEdge, make_layer_map
from cute import Cute
from master import Master
from worker import Worker
from dataset import Dataset
from program_arguments import ProgramArguments

def gpu_fix():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def graph_genome_main(args: List[str]):
    gpu_fix()
    
    genome_path = args[2]
    image_dst = args[3]

    genome: CnnGenome = pickle.load(open(genome_path, 'rb'))

    model = genome.create_model()

    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file=image_dst,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )


def make_example_genome():
    # The first argument to all layer and edge constructors is an innovation
    # number, hardcoded in this case

    # Using the MNIST dataset (28x28 images with a single channel (grayscale))
    input_layer = InputLayer(0, 28, 28, 1)

    ## Out intermediate layers
    # Layer 1 will have a volume size of 18x18x64
    layer_1 = Layer(1, 18, 18, 64)
    
    # Layer 3 is a pooling layer so the number of channels is the same as the
    # number of input channels in this case, we are connecting it directly to
    # the input layer so it will have 1 output channel. 
    # We will use a volume size of 8x8x1
    layer_3 = Layer(3, 8, 8, 1)

    # 10 output classes for the 10 digits. We will have two dense layers
    # (asside from the classification layer) each with 128 units
    output_layer = OutputLayer(2, [128, 128], 10)
    
    # This is used to make serialization go smoothly.
    # Maps innovation number to layer object.
    layer_map = make_layer_map([input_layer, layer_1, layer_3, output_layer])

    # We choose a stride of 1, supply the input layer and output layer, and the
    # filter size will be compted along with the required number of filters.
    edge_0 = SeparableConvEdge(0, 1, input_layer.layer_innovation_number, 
            layer_1.layer_innovation_number, layer_map)

    # A dense edge - an edge that is connected to the output layer
    edge_1 = DenseEdge(1, layer_1.layer_innovation_number,
            output_layer.layer_innovation_number, layer_map)

    # Our pooling edge - we just need to give it the input and output layers
    edge_2 = FractionalMaxPoolingEdge(2, input_layer.layer_innovation_number,
            layer_3.layer_innovation_number, layer_map)
    
    # A dense edge connecting layer_3 to the output
    edge_3 = DenseEdge(3, layer_3.layer_innovation_number,
            output_layer.layer_innovation_number, layer_map)

    # Connecting layer_1 to layer_3 using a Conv2D operation - use a stride of 2
    edge_4 = ConvEdge(4, 2, layer_1.layer_innovation_number,
            layer_3.layer_innovation_number, layer_map)

    # Redundant spesification of 10 output classes - this will be refactored
    genome = CnnGenome(10, input_layer, output_layer, layer_map,
            [edge_0, edge_2, edge_4], [edge_1, edge_3], {}, set(), set())

    pickle.dump(genome, open("test_genome.cnn_genome", "bw"))
    graph_genome_main([0, 0, "test_genome.cnn_genome", "test_genome.png"])    
    


def train_genome_main(args: List[str]):
    gpu_fix()
    hp.set_dataset(Dataset.make_mnist_dataset())
    genome_path = args[2]
    
    genome: CnnGenome = pickle.load(open(genome_path, 'rb'))

    genome.train()


def evo_main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.DEBUG, format=f'[%(asctime)s][rank {rank}] %(message)s')
    
    pa = ProgramArguments()
    
    gpu_fix()

    if rank == 0:
        max_rank: int = comm.Get_size()

        cute: Cute = Cute(pa)
        master = Master(cute, comm, max_rank)
        master.run()
    else:
        worker = Worker(rank, comm)
        worker.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    if sys.argv[1] == "graph_genome":
        graph_genome_main(sys.argv)
    elif sys.argv[1] == "train_genome":
        train_genome_main(sys.argv)
    elif sys.argv[1] == "make_example_genome":
        make_example_genome()
    else:
        evo_main()
