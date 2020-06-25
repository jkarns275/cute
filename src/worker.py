import logging

from mpi4py import MPI
import tensorflow as tf

from examm import EXAMM
from cnn.cnn_genome import CnnGenome
import tags
import requests


class Worker:


    def __init__(self, rank: int, comm: MPI.Intracomm):
        self.rank: int = rank
        self.comm: MPI.Intracomm = comm
        self.done: bool = False
        
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logginf.info(f"found {len(gpus)} gpus, and {len(logical_gpus)} physical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def run(self):
        handlers = {
            tags.TERMINATE_TAG: self.handle_terminate,
            tags.GENOME_TAG: self.handle_genome
        }
        
        logging.info(f"worker {self.rank} is beginning")

        while not self.done:
            logging.debug(f"worker {self.rank} sending work request")
            requests.send_work_request(self.comm, 0)

            status = MPI.Status()
            self.comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)

            tag: int = status.Get_tag()
            logging.debug(f"worker probe recieved message with tag {tag}")

            if tag in handlers:
                handler = handlers[tag]
                handler()
            else:
                logging.fatal(f"recieved unrecognized tag {tag} from {source}")
                self.comm.Abort(1)
        
        logging.info(f"worker {self.rank} terminating")


    def handle_terminate(self):
        logging.debug(f"worker {self.rank} handling terminate request")
        self.done = True


    def handle_genome(self):
        logging.debug(f"worker {self.rank} handling genome")

        genome: CnnGenome = requests.recieve_genome(self.comm, 0)
        genome.train()

        requests.send_genome(self.comm, 0, genome)


