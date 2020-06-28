import os
import sys
import pickle
import logging
from typing import List
# This hides tensorflow debug output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

from mpi4py import MPI
import tensorflow as tf

import hp
from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer
from examm import EXAMM
from master import Master
from worker import Worker
from datasets import Dataset
from program_arguments import ProgramArguments

def gpu_fix():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f"found {len(gpus)} gpus, and {len(logical_gpus)} physical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def graph_genome_main(args: List[str]):
    
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


def train_genome_main(args: List[str]):
    hp.set_dataset(Dataset.make_mnist_dataset())
    genome_path = args[2]
    
    genome: CnnGenome = pickle.load(open(genome_path, 'rb'))

    genome.train()


def evo_main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.DEBUG, format=f'[%(asctime)s][rank {rank}] %(message)s')
    
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
    logging.getLogger().setLevel(logging.DEBUG)

    gpu_fix()

    if sys.argv[1] == "graph_genome":
        graph_genome_main(sys.argv)
    elif sys.argv[1] == "train_genome":
        train_genome_main(sys.argv)
    else:
        evo_main()
