import time
import pickle
import logging

import numpy as np

from cnn import CnnGenome, Edge, ConvEdge, DenseEdge, Layer, InputLayer, OutputLayer, make_layer_map
from program_arguments import ProgramArguments
from speciation_strategy import SpeciationStrategy
from speciation_strategy.island_speciation_strategy import IslandSpeciationStrategy
from fitness_log import FitnessLog
import hp

class Cute:
    

    def __init__(self, program_arguments: ProgramArguments):
        self.population_size: int = program_arguments.args.population_size
        self.number_islands: int = program_arguments.args.number_islands
        self.bp_iterations: int = program_arguments.args.backprop_iterations
        self.max_genomes: int = program_arguments.args.max_genomes
        self.output_directory: str = program_arguments.args.output_directory[0]

        self.fitness_log: FitnessLog = FitnessLog(self.output_directory)

        initial_genome: CnnGenome = self.generate_initial_genome()
 
        self.speciation_strategy: SpeciationStrategy = \
            IslandSpeciationStrategy(initial_genome, self.number_islands, self.population_size)

        # For the initial genome
        self.speciation_strategy.generated_genomes += 1
        
        self.mutation_function_probability_map = {
                CnnGenome.add_edge_mut:                 hp.add_edge_probability,
                CnnGenome.add_layer_mut:                hp.add_layer_probability,
                CnnGenome.disable_edge_mut:             hp.disable_edge_probability,
                CnnGenome.enable_edge_mut:              hp.enable_edge_probability,
                CnnGenome.disable_layer_mut:            hp.disable_layer_probability,
                CnnGenome.enable_layer_mut:             hp.enable_layer_probability,
                CnnGenome.copy_mut:                     hp.copy_probability
            }
        
        self.mutation_functions = []
        self.mutation_probabilities = []

        for mutation_function, probability in self.mutation_function_probability_map.items():
            self.mutation_functions.append(mutation_function)
            self.mutation_probabilities.append(probability)

        self.rng: np.random.Generator = np.random.Generator(np.random.PCG64(int(str(time.time()).split('.')[1])))
        _warmup = self.rng.random(1000)


    def generate_initial_genome(self):
        input_layer: InputLayer = InputLayer(Layer.get_next_layer_innovation_number(), 28, 28, 1)
        hidden_layer: Layer = Layer(Layer.get_next_layer_innovation_number(), 8, 8, 16)
        output_layer: OutputLayer = OutputLayer(Layer.get_next_layer_innovation_number(), [128, 32], 10)

        layer_map = make_layer_map([input_layer, hidden_layer, output_layer])

        edge_1: ConvEdge = ConvEdge(Edge.get_next_edge_innovation_number(), 2, input_layer.layer_innovation_number,
                                    hidden_layer.layer_innovation_number, layer_map)
        edge_2: DenseEdge = DenseEdge(Edge.get_next_edge_innovation_number(), hidden_layer.layer_innovation_number,
                                        output_layer.layer_innovation_number, layer_map)
        
        genome = CnnGenome(10, input_layer, output_layer, layer_map, [edge_1], [edge_2], {}, set(), set())
       
        logging.info("performing some tests of CnnGenome::path_exists")

        assert genome.path_exists(input_layer, output_layer)
        assert genome.path_exists(input_layer, hidden_layer)
        assert not genome.path_exists(output_layer, input_layer)
        assert not genome.path_exists(hidden_layer, input_layer)

        return genome


    def unimplemented(self, m: str):
        logging.debug(f"called unimplemented method 'Cute::{m}'")
        # raise Exception(f"method 'Cute::{m}' has not been implemented")

    
    def update_logs(self):
        self.unimplemented('update_logs')


    def generate_genome(self, only_mutate=False):
        if self.get_generated_genomes() >= self.max_genomes:
            return None
        
        self.speciation_strategy.generated_genomes += 1
        
        if not only_mutate and self.rng.random() < hp.co_rate:
            # Intra island crossover
            if self.rng.random() * hp.co_rate < hp.intra_island_co_rate:
                logging.info("trying intra island crossover")
                parent_fn = self.speciation_strategy.try_get_intra_island_crossover_parents
            else:
                logging.info("trying inter island crossover")
                parent_fn = self.speciation_strategy.try_get_inter_island_crossover_parents

            while True:
                parents = parent_fn(self.rng)
                if not parents:
                    logging.info("failed to perform crossover (couldn't acquire parents)")
                    return self.generate_genome(only_mutate=True)

                child: Optional[CnnGenome] = CnnGenome.try_crossover(self.rng, *parents)
                
                if child:
                    return child
        else:
            # Grab genome from speciation strategy
            genome: CnnGenome = self.speciation_strategy.generate_genome(self.rng)
            
            # Keep trying random mutations until one succeeds
            while True:
                mutation_function = self.rng.choice(self.mutation_functions, 1, p=self.mutation_probabilities)[0]

                if mutation_function(genome, self.rng):
                    break

            return genome
    

    def try_insert_genome(self, genome: CnnGenome):
        insertion_type: str = self.speciation_strategy.try_insert_genome(genome)

        if insertion_type == "new_best":
            best_genome: CnnGenome = self.speciation_strategy.get_best_genome()
            pickle.dump(best_genome, open(f"{self.output_directory}/{self.get_generated_genomes()}.cnn_genome", "bw"))
            print("\n" + \
                  "                                                    _,,_..                   \n" + \
                  "                                             _,,--''  '':'::!...             \n" + \
                  "  new global best kiwi                   _,-' ...::::::' ' ' '::::.          \n" + \
                  "                                      _.' .':'''..'... . '::' ':::::.        \n" + \
                  "                        .,...____,,.-'  ..''  .:.'' ...':''.:'''':::::       \n" + \
                  "                     ,.:::'''''.`'^     ' ..'' ...'' ':..''.''.   '::::      \n" + \
                  "                    .:::''''  ''   ':  .''....:.....':'':''.''.  ''::::.     \n" + \
                  "                    '''          ... .  ''''.''''''...:'.'. '..'. . ::::     \n" + \
                  "                    :  O          .. '.    ..'...    ''. '.'.  : '::.:::     \n" + \
                  "                    :'           . '.. ''.. :..  ''.   .'..'.: ' ..;::::     \n" + \
                  "                    :\" __,..---- .:i  '    '.'.''...:'. ''.:.:'  :::: :     \n" + \
                  "                  .:'.-'            `.... .. '.''.  '''':. : ::  :';::'      \n" + \
                  "                .:'.:'                '::::::..:':::..:. '. ': ::':;.'       \n" + \
                  "               ::.''                    '':::::::''''.:'::.. ...::::;        \n" + \
                  "             .::'                           '''''::::'''''::.::'::::         \n" + \
                  "           .::'                                            :::'...:          \n" + \
                  "          ::'                                             ;::  .:;           \n" + \
                  "         :'                                              ,::' ,:.:           \n" + \
                  "       .:'                                          '':::::: .:::            \n" + \
                  "     .:'                                         ..:''''.'  .:::             \n" + \
                  "    :'                                                 ''':::::'             \n" + \
                  "   :'                                                ..''' .'                \n" + \
                  "  '                                                     .''                  \n" + \
                  "                                                       '                     \n")

        self.fitness_log.update_log(self)


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

    
    def get_best_accuracy(self):
        return self.speciation_strategy.get_best_accuracy()


    def get_worst_accuracy(self):
        return self.speciation_strategy.get_worst_accuracy()
