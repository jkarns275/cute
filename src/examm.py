import time
import logging

import numpy as np

from program_arguments import ProgramArguments
from speciation_strategy import SpeciationStrategy
from speciation_strategy.island_speciation_strategy import IslandSpeciationStrategy
from cnn import CnnGenome, Edge, ConvEdge, DenseEdge, Layer, InputLayer, OutputLayer, make_layer_map


class EXAMM:
    
    ## When performing crossover, what proportion of the time nodes and edges from the more fit parent should 
    # be put in the resulting child genome.
    more_fit_crossover_rate: float  = 1.0

    ## When performing crossover, what proportion of the time nodes and edges from the less fit parent should
    # be put in the resulting child genome.
    less_fit_crossover_rate: float  = 0.50

    # So these rates aren't quite proportions: the sum of them is the denominator, and the proportion
    # is the fraction of a given rate and that sum.

    ## How often the add edge mutation should be performed
    add_edge_rate: float            = 1.0
    ## How often the enable edge mutation should be performed
    enable_edge_rate: float         = 1.0
    ## How often the disable edge mutation should be performed
    disable_edge_rate: float        = 1.0
    ## How often the split edge mutation should be performed
    split_edge_rate: float          = 1.0
    ## How often the clone mutation should be performed
    clone_rate: float               = 1.0
   

    def __init__(self, program_arguments: ProgramArguments):
        self.population_size: int = program_arguments.args.population_size
        self.number_islands: int = program_arguments.args.number_islands
        self.bp_iterations: int = program_arguments.args.backprop_iterations
        self.max_genomes: int = program_arguments.args.max_genomes
        self.output_directory: str = program_arguments.args.output_directory

        initial_genome: CnnGenome = self.generate_initial_genome()

        self.mutation_rate: float = 0.7
        self.intra_island_co_rate: float = 0.2
        self.inter_island_co_rate: float = 0.1

        self.speciation_strategy: SpeciationStrategy = \
            IslandSpeciationStrategy(initial_genome, self.number_islands, self.population_size,
                                    self.mutation_rate, self.intra_island_co_rate, self.inter_island_co_rate)

        self.rng: np.random.Generator = np.random.Generator(np.random.PCG64(int(str(time.time()).split('.')[1])))


    def generate_initial_genome(self):
        input_layer: InputLayer = InputLayer(Layer.get_next_layer_innovation_number(), 28, 28, 1)
        print(type(input_layer))
        hidden_layer: Layer = Layer(Layer.get_next_layer_innovation_number(), 14, 14, 14)
        output_layer: OutputLayer = OutputLayer(Layer.get_next_layer_innovation_number(), [64], 10)

        layer_map = make_layer_map([input_layer, hidden_layer, output_layer])

        edge_1: ConvEdge = ConvEdge(Edge.get_next_edge_innovation_number(), 1, input_layer.layer_innovation_number,
                                    hidden_layer.layer_innovation_number, layer_map)
        edge_2: DenseEdge = DenseEdge(Edge.get_next_edge_innovation_number(), hidden_layer.layer_innovation_number,
                                        output_layer.layer_innovation_number, layer_map)
        
        genome = CnnGenome(10, input_layer, output_layer, layer_map, [edge_1], [edge_2])
        
        return genome


    def unimplemented(self, m: str):
        logging.debug(f"called unimplemented method 'EXAMM::{m}'")
        # raise Exception(f"method 'EXAMM::{m}' has not been implemented")

    
    def update_logs(self):
        self.unimplemented('update_logs')


    def generate_genome(self):
        if self.get_generated_genomes() >= self.max_genomes:
            return None
        
        genome = self.speciation_strategy.generate_genome(self)
        
        self.unimplemented('generate_genome')

        return genome
    

    def try_insert_genome(self, genome: CnnGenome):
        self.speciation_strategy.try_insert_genome(genome)


    def mutate(self, n_mutations: int, genome: CnnGenome):
        self.unimplemented('mutate')
        return genome


    def crossover(self, better_parent: CnnGenome, worse_parent: CnnGenome):
        self.unimplemented('crossover')
        return better_parent


    def get_inserted_genomes(self):
        return self.speciation_strategy.inserted_genomes


    def get_generated_genomes(self):
        return self.speciation_strategy.generated_genomes


    def get_worst_genome(self):
        return self.speciation_strategy.get_worst_genome()


    def get_best_genome(self):
        return self.speciation_strategy.get_best_genome()


    def get_best_fitness(self):
        return self.speciation_strategy.get_best_fitness()


    def get_worst_fitness(self):
        return self.speciation_strategy.get_worst_fitness()


