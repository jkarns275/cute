from mpi4py import MPI

import tags
from cnn.cnn_genome import CnnGenome


def recieve_work_request(comm: MPI.Intercomm, source: int):
    data = comm.recv(source=source, tag=tags.WORK_REQUEST_TAG)
    return data['message']


def send_work_request(comm: MPI.Intercomm, dest: int):
    data = { 'message': 0 }
    comm.send(data, dest=dest, tag=tags.WORK_REQUEST_TAG)


def recieve_genome(comm: MPI.Intercomm, source: int):
    genome = comm.recv(source=source, tag=tags.GENOME_TAG)
    return genome


def send_genome(comm: MPI.Intercomm, dest: int, genome: CnnGenome):
    comm.send(genome, dest=dest, tags=tags.GENOME_TAG)


def recieve_terminate(comm: MPI.Intercomm, source: int):
    data = comm.recv(source=source, tags=tags.TERMINATE_TAG)
    return data


def send_terminate(comm: MPI.Intercomm, dest: int):
    data = { 'message': 0 }
    comm.send(dest=dest, tags=tags.TERMINATE_TAG)
