import math

from util import sorted_insert
from cnn import CnnGenome


class Island:

    def __init__(self, population_size: int):
        self.population = []
        self.population_size = population_size

    
    def try_insert_genome(self, genome: CnnGenome):
        if len(self.population) < self.population_size:
            self.population.append(genome)
            return None, 0

        elif self.population[-1].mse > genome.mse:
            insert_position = sorted_insert(genome, self.population, key=lambda genome: genome.mse)
            genome = self.population.pop()

            # Sanity check.
            assert insert_position < self.population_size

            return genome, insert_position


    def get_best_genome(self):
        if len(self.population) == 0:
            return None

        return self.population[0]


    def get_best_fitness(self):
        if len(self.population) == 0:
            return math.inf
        
        return self.get_best_genome().mse


    def get_worst_genome(self):
        if len(self.population) == 0:
            return None

        return self.population[-1].mse


    def get_worst_fitness(self):
        if len(self.population) == 0:
            return math.inf

        return self.get_worst_genome().mse


    def is_full(self):
        return len(self.population) == self.population_size
