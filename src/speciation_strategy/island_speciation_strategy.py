import logging
from typing import List

from speciation_strategy import SpeciationStrategy
from cnn import CnnGenome
from speciation_strategy.island import Island

class IslandSpeciationStrategy(SpeciationStrategy):

    def __init__(self, initial_genome: CnnGenome, number_islands: int, population_size: int,
                mutation_rate: float, inter_island_crossover_rate: float, intra_island_crossover_rate: float):
        super().__init__()

        self.population_size = population_size
        self.number_islands = number_islands

        self.islands: List[Island] = list(map(lambda _: Island(population_size), range(number_islands)))
        self.global_best_genome = initial_genome
        self.global_worst_genome = initial_genome

        self.mutation_rate: float = mutation_rate
        self.inter_island_crossover_rate: float = inter_island_crossover_rate
        self.intra_island_crossover_rate: float = intra_island_crossover_rate

        # Make sure these rates sum to 1.0
        assert 0.9999 < mutation_rate + inter_island_crossover_rate + intra_island_crossover_rate < 1.00001
        

    def try_insert_genome(self, genome: CnnGenome):
        """
        Attempts to insert the supplied genome.
        If the genome is inserted, this method will return True, otherwise it will return False.
        """
        logging.debug("called disabled method IslandSpeciationStrategy")
        return True

        if self.global_best_genome.mse > genome.mse:
            self.global_best_genome = genome
        
        removed_genome, insert_position  = self.islands[genome.island].try_insert_genome(genome)
        inserted = removed_genome != None and insert_position >= 0

        if not inserted:
            return False
        elif removed_genome == self.global_worst_genome and insert_position == self.global_worst_genome:
            self.global_worst_genome = genome
        
            self.inserted_genomes += 1

            return True


    def get_best_fitness(self):
        logging.debug("called abstract get_best_fitness")        
        # raise Exception("Called abstract get_best_fitness")


    def get_worst_fitness(self):
        logging.debug("called abstract get_worst_fitness")
        # raise Exception("Called abstact get_worst_fitness")


    def get_generated_genomes(self):
        return self.generated_genomes


    def get_inserted_genomes(self):
        return self.inserted_genomes


    def get_global_best_genome(self):
        logging.debug("called abstract get_global_best_genome")
        # raise Exception("Called abstract get_global_best_genome")


    def get_global_worst_genome(self):
        logging.debug("called abstract get_global_worst_genome")
        # raise Exception("Called abstract get_global_worst_genome")

    def generate_genome(self, examm: 'EXAMM'):
        logging.debug("called abstract generate_genome")
        self.generated_genomes += 1
        return CnnGenome()
