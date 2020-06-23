import argparse

from datasets import Dataset
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

        self.args = self.parse_args()

        self.set_dataset()

    
    def set_dataset(self):
        self.args.dataset = self.args.dataset[0].lower()

        dataset = Dataset.dataset_from_arguments(self)
        set_dataset(dataset)
