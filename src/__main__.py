import logging
import sys

from mpi4py import MPI

from master import Master
from worker import Worker
from program_arguments import ProgramArguments
from examm import EXAMM


def test_network():
    from cnn import ConvEdge, CnnGenome, DenseEdge, Edge, Layer, InputLayer, OutputLayer
    
    input_layer: InputLayer = InputLayer(0, 28, 28, 1)
    hidden_layer: Layer = Layer(1, 14, 14, 14)
    output_layer: OutputLayer = OutputLayer(2, [64], 10)

    edge_1: ConvEdge = ConvEdge(0, 1, input_layer, hidden_layer)
    edge_2: DenseEdge = DenseEdge(1, hidden_layer, output_layer)
    
    genome = CnnGenome(10, input_layer, output_layer, [hidden_layer], [edge_1], [edge_2])
    genome.create_model()

test_network()
sys.exit(0)


def main(pa: ProgramArguments, comm: MPI.Intracomm, rank: int):
    if rank == 0:
        max_rank: int = comm.Get_size()

        examm: EXAMM = EXAMM(pa)
        master = Master(examm, comm, max_rank)
        master.run()
    else:
        worker = Worker(rank, comm)
        worker.run()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pa = ProgramArguments()
    
    logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s][rank {rank}] %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    main(pa, comm, rank)
