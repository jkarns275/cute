from mpi4py import MPI
import logging

from examm import EXAMM
from cnn.cnn_genome import CnnGenome
import tags
import requests


class Master:


    def __init__(self, examm: EXAMM, comm: MPI.Intracomm, max_rank: int):
        self.examm: EXAMM = examm
        self.comm: MPI.Intracomm = comm
        self.max_rank: int = max_rank
        self.terminates_sent: int = 0


    def run(self):
        handlers = {
            tags.WORK_REQUEST_TAG:  self.handle_work_request,
            tags.GENOME_TAG:        self.handle_genome,
        }
        
        logging.info("starting master loop")

        while not self.done():
            status = MPI.Status()
            self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    
            source = status.Get_source()
            tag = status.Get_tag()
            logging.debug(f"probe returned message from {source} with tag {tag}")

            if tag in handlers:
                handler = handlers[tag]
                handler(source)
            else:
                logging.fatal(f"recieved unrecognized tag {tag} from {source}")
                self.comm.Abort(1)
        
        logging.info("ending master")


    def done(self):
        return self.terminates_sent >= self.max_rank - 1


    def handle_work_request(self, source: int):
        logging.debug(f"handling work request from {source}")
        work_request_message = requests.recieve_work_request(self.comm, source)
        
        genome: CnnGenome = self.examm.generate_genome()

        if genome is None:
            logging.debug(f"terminating worker {source}")

            requests.send_terminate(self.comm, source)            
            self.terminates_sent += 1
        else:
            logging.debug(f"sending genome to {source}")
            requests.send_genome(self.comm, source, genome)


    def handle_genome(self, source: int):
        logging.debug(f"handling genome from {source}")
        genome: CnnGenome = requests.recieve_genome(self.comm, source)
        logging.debug(f"recieved genome with fitness of {genome.fitness}")
        self.examm.try_insert_genome(genome)
