import logging
from typing import List, Optional, Tuple

import numpy as np

from speciation_strategy import SpeciationStrategy
from cnn import CnnGenome
from speciation_strategy.island import Island
if False:
    from examm import EXAMM


class IslandSpeciationStrategy(SpeciationStrategy):


    def __init__(self, initial_genome: CnnGenome, number_islands: int, population_size: int):
        super().__init__()

        self.population_size = population_size
        self.number_islands = number_islands

        self.initial_genome: CnnGenome = initial_genome
        
        self.islands: List[Island] = list(map(lambda _: Island(population_size), range(number_islands)))
        self.global_best_genome = initial_genome
        self.global_worst_genome = initial_genome

        # We need to rotate through islands 0 through n - 1, this is the counter we'll use
        self.island_turn: int = 0


    def try_insert_genome(self, genome: CnnGenome) -> Optional[str]:
        """
        Attempts to insert the supplied genome.
        If the genome is inserted, this method will return "inserted", otherwise it will return None
        If the genome is the new global best it will return "new_best"
        """

        if self.global_best_genome.fitness > genome.fitness:
            self.global_best_genome = genome
            new_best = True
        else:
            new_best = False
        
        removed_genome, insert_position  = self.islands[genome.island].try_insert_genome(genome)
        inserted = removed_genome != None or insert_position >= 0
        
        if not inserted:
            return None
        
        self.inserted_genomes += 1
        
        # If this is the new global worst genome...
        if removed_genome == self.global_worst_genome and insert_position == self.population_size - 1:
            self.global_worst_genome = genome
            
        return "new_best" if new_best else "inserted"


    def get_best_fitness(self):
        return self.global_best_genome.fitness


    def get_worst_fitness(self):
        return self.global_worst_genome.fitness


    def get_generated_genomes(self):
        return self.generated_genomes


    def get_inserted_genomes(self):
        return self.inserted_genomes


    def get_best_genome(self):
        return self.global_best_genome


    def get_worst_genome(self):
        return self.global_worst_genome


    def get_best_accuracy(self):
        return self.global_best_genome.accuracy

    
    def get_worst_accuracy(self):
        return self.global_worst_genome.accuracy

    
    def next_island_turn(self):
        # this could be replaced with modulus
        turn = self.island_turn
        
        self.island_turn += 1
        if self.island_turn == self.number_islands:
            self.island_turn = 0
        
        return turn

    def peek_next_island_turn(self):
        return self.island_turn


    def try_get_inter_island_crossover_parents(self, rng: np.random.Generator) -> Optional[Tuple[CnnGenome, ...]]:
        # peek since we may not actually be able to give this island a turn (i.e. get a genome from this island
        # and another island)
        island_turn = self.peek_next_island_turn()

        if self.islands[island_turn].population:
            # islands with a population that we can grab a genome from
            valid_islands = []
            for i, island in enumerate(self.islands):
                if i == island_turn:
                    continue
                if island.population:
                    valid_islands.append(island)
            
            if not valid_islands:
                return None 

            population_0 = self.islands[island_turn].population
            population_1 = valid_islands[rng.integers(0, len(valid_islands))].population

            i0 = rng.integers(0, len(population_0))
            i1 = rng.integers(0, len(population_1))

            g0 = population_0[i0]
            g1 = population_1[i1]
            
            # we did use the peeked island, so this will increment the island turn
            self.next_island_turn()

            return (g0, g1)
        else:
            return None


    def try_get_intra_island_crossover_parents(self, rng: np.random.Generator) -> Optional[Tuple[CnnGenome, ...]]:
        # Peek since we may or may not actually give this island a turn (i.e. actually get parent genomes from it)
        island_turn = self.peek_next_island_turn()

        if len(self.islands[island_turn].population) < 2:
            return None
        else:
            population: List[CnnGenome] = self.islands[island_turn].population
            i0 = rng.integers(0, len(population))
            i1 = rng.integers(0, len(population) - 1)
            
            if i1 == i0:
                i1 += 1
            
            assert i1 != i0
            
            # we did use the peeked island turn so this will increment the island turn
            self.next_island_turn()

            return (population[i0], population[i1])


    def generate_genome(self, rng: np.random.Generator):
        island_turn = self.next_island_turn()
        
        if self.inserted_genomes == 0:
            genome = self.initial_genome.copy()
            genome.island = island_turn
            return genome
        else:
            
            # This should only happen during the beginning of the program
            if self.islands[island_turn].is_empty():
                genome = self.initial_genome
            else:
                genome = self.islands[island_turn].get_random_genome(rng)
                
            genome.island = island_turn
            
            genome = genome.copy()

            return genome

