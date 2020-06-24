import math
import logging
from typing import List

import numpy as np

from util import sorted_insert
from cnn import CnnGenome


class Island:

    def __init__(self, population_size: int):
        self.population: List[CnnGenome] = []
        self.population_size: int = population_size

    
    def try_insert_genome(self, genome: CnnGenome):
        logging.info(f"inserting genome with fitness of {genome.fitness}")
        
        if len(self.population) < self.population_size:
            self.population.append(genome)
            return None, 0
        elif self.population[-1].fitness > genome.fitness:
            insert_position = sorted_insert(genome, self.population, key=lambda genome: genome.fitness)
            genome = self.population.pop()

            # Sanity check.
            assert insert_position < self.population_size

            return genome, insert_position
        else:
            return None, -1

    
    def get_random_genome(self, rng: np.random.Generator):
        if self.population:
            index = rng.integers(0, len(self.population))
            return self.population[index]


    def get_best_genome(self):
        if len(self.population) == 0:
            return None

        return self.population[0]


    def get_best_fitness(self):
        if len(self.population) == 0:
            return math.inf
        
        return self.get_best_genome().fitness


    def get_worst_genome(self):
        if len(self.population) == 0:
            return None

        return self.population[-1].fitness


    def get_worst_fitness(self):
        if len(self.population) == 0:
            return math.inf

        return self.get_worst_genome().fitness


    def is_full(self):
        return len(self.population) == self.population_size


    def is_empty(self):
        return len(self.population) == 0
