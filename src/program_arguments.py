import os
import argparse

import tensorflow as tf

from dataset import Dataset
from hp import set_dataset
import hp


class ProgramArguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__(description='EXAMM for TensorFlow', add_help=True)

        self.dataset: str = None
        self.output_directory: str = None
        self.number_islands: int = None
        self.population_size: int = None
        self.max_genomes: int = None
        self.backprop_iterations: int = None

        self.add_argument('dataset', metavar='dataset', type=str, nargs=1,
                            help='the image dataset to be used, select one from the available datasets here: https://www.tensorflow.org/datasets/catalog/overview')
        self.add_argument('output_directory', metavar='output_directory', type=str, nargs=1,
                            help='the directory output logs should be stored in')
        self.add_argument('-ni', '--number_islands', metavar='number_islands', action='store',
                default=1, type=int, help='the number of separate islands to use')
        self.add_argument('-ps', '--population_size', metavar='population_size', action='store',
                default=10, type=int, help='the maximum number of genomes on each islands')
        self.add_argument('-mg', '--max_genomes', metavar='max_genomes', action='store',
                default=1000, type=int, help='the number of genomes to generate and evaluate')
        self.add_argument('-bpi', '--backprop_iterations', metavar='backprop_iterations', action='store',
                default=1, type=int, help='the number of iterations of backpropagation to be applied to generated genomes to evaluate them')
        self.add_argument('-ig', '--ignore_gpus', metavar='ignore_gpus', action='store',
                default=0, type=int, help='whether or not to ignore gpus. if set cpus will be used instead')

        self.args = self.parse_args()

        self.set_dataset()
        self.set_number_epochs()
        self.set_ignore_gpus()


    def set_number_epochs(self):
        hp.set_number_epochs(self.args.backprop_iterations)
   

    def set_dataset(self):
        self.args.dataset = self.args.dataset[0].lower()

        dataset = Dataset.dataset_from_arguments(self)
        set_dataset(dataset)


    def set_ignore_gpus(self):
        if self.args.ignore_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            tf.config.experimental.set_visible_devices([], 'GPU')
