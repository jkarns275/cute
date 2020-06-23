import logging
import sys

from mpi4py import MPI

from master import Master
from worker import Worker
from program_arguments import ProgramArguments
from examm import EXAMM


def make_genome():
    from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer
    import tensorflow as tf
    from cnn.cnn_util import make_layer_map

    input_layer: InputLayer = InputLayer(0, 28, 28, 1)
    hidden_layer: Layer = Layer(1, 14, 14, 14)
    output_layer: OutputLayer = OutputLayer(2, [64], 10)

    layer_map = make_layer_map([input_layer, hidden_layer, output_layer])

    edge_1: ConvEdge = ConvEdge(0, 1, input_layer.layer_innovation_number, hidden_layer.layer_innovation_number, layer_map)
    edge_2: DenseEdge = DenseEdge(1, hidden_layer.layer_innovation_number, output_layer.layer_innovation_number, layer_map)
    
    genome = CnnGenome(10, input_layer, output_layer, layer_map, [edge_1], [edge_2])
    
    return genome
    # model = genome.create_model()

    # model.summary()

    # tf.keras.utils.plot_model(
    #     model,
    #     to_file="model.png",
    #     show_shapes=False,
    #     show_layer_names=True,
    #     rankdir="TB",
    #     expand_nested=False,
    #     dpi=96,
    # )


def main(pa: ProgramArguments, comm: MPI.Intracomm, rank: int):
    if rank == 0:
        max_rank: int = comm.Get_size()

        examm: EXAMM = EXAMM(pa)
        master = Master(examm, comm, max_rank, make_genome)
        master.run()
    else:
        worker = Worker(rank, comm)
        worker.run()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pa = ProgramArguments()
    
    logging.basicConfig(level=logging.DEBUG, format=f'[%(asctime)s][rank {rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)

    main(pa, comm, rank)
