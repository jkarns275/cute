import logging
import sys

import os
# This hides tensorflow debug output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from mpi4py import MPI

from master import Master
from worker import Worker
from program_arguments import ProgramArguments
from examm import EXAMM


def make_genome():
    from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer
    import tensorflow as tf
    from cnn.cnn_util import make_layer_map

#     model = genome.create_model()
# 
#     model.summary()
# 
#     tf.keras.utils.plot_model(
#         model,
#         to_file="model.png",
#         show_shapes=False,
#         show_layer_names=True,
#         rankdir="TB",
#         expand_nested=False,
#         dpi=96,
#     )


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
    import hp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.DEBUG, format=f'[%(asctime)s][rank {rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    
    pa = ProgramArguments()
    main(pa, comm, rank)
