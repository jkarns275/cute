from typing import List

from examm import EXAMM
from speciation_strategy import SpeciationStrategy
from cnn import CnnGenome
from speciation_strategy import Island

class IslandSpeciationStrategy(SpeciationStrategy):

    def __init__(self, initial_genome: CnnGenome, number_islands: int, population_size: int):
        super().__init__()

        self.population_size = population_size
        self.number_islands = number_islands

        self.islands: List[Island] = list(map(lambda _: Island(population_size), range(number_islands)))
        self.global_best_genome = initial_genome
        self.global_worst_genome = initial_genome

    def try_insert_genome(self, genome: CnnGenome):
        """
        Attempts to insert the supplied genome.
        If the genome is inserted, this method will return True, otherwise it will return False.
        """
        
        if self.global_best_genome.mse > genome.mse:
            self.global_best_genome = genome
        
        removed_genome, insert_position  = self.islands[genome.island].try_insert_genome(genome)
        inserted = removed_genome != None && insert_position >= 0

        if not inserted:
            return False
        elif removed_genome == self.global_worst_genome and insert_position == self.global_worst_genome:
            self.global_worst_genome = genome

        return inserted


    def get_best_fitness(self):
        raise Exception("Called abstract get_best_fitness")


    def get_worst_fitness(self):
        raise Exception("Called abstact get_worst_fitness")


    def get_generated_genomes(self):
        raise Exception("Called abstract get_generated_genomes")


    def get_inserted_genomes(self):
        raise Exception("Called abstract get_inserted_genomes")


    def get_global_best_genome(self):
        raise Exception("Called abstract get_global_best_genome")


    def get_global_worst_genome(self):
        raise Exception("Called abstract get_global_worst_genome")

    def generate_genome(self, examm: EXAMM):
        raise Exception("Called abstract generate_genome")

