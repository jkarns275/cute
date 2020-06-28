import os
import sys
import pickle
import logging
from typing import List
# This hides tensorflow debug output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

from mpi4py import MPI
import tensorflow as tf

from master import Master
from worker import Worker
from program_arguments import ProgramArguments
from examm import EXAMM


def graph_genome_main(args: List[str]):
    from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer
    from cnn.cnn_util import make_layer_map

    genome_path = args[2]
    image_dst = args[3]

    genome: CnnGenome = pickle.load(open(genome_path, 'rb'))

    model = genome.create_model()

    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file=image_dst,
        show_shapes=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )


def evo_main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.DEBUG, format=f'[%(asctime)s][rank {rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    
    pa = ProgramArguments()

    if rank == 0:
        max_rank: int = comm.Get_size()

        examm: EXAMM = EXAMM(pa)
        master = Master(examm, comm, max_rank, make_genome)
        master.run()
    else:
        worker = Worker(rank, comm)
        worker.run()


if __name__ == "__main__":
    import hp
    

    if sys.argv[1] == "graph_genome":
        graph_genome_main(sys.argv)
    else:
        evo_main()
