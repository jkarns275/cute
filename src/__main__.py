import logging
from mpi4py import MPI

from master import Master
from worker import Worker
from program_arguments import ProgramArguments
from examm import EXAMM

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
    
    main(pa, comm, rank)
