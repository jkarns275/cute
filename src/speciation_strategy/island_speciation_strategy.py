import logging
from typing import List

from speciation_strategy import SpeciationStrategy
from cnn import CnnGenome
from speciation_strategy.island import Island
if False:
    from examm import EXAMM

class IslandSpeciationStrategy(SpeciationStrategy):

    def __init__(self, initial_genome: CnnGenome, number_islands: int, population_size: int,
                mutation_rate: float, inter_island_crossover_rate: float, intra_island_crossover_rate: float):
        super().__init__()

        self.population_size = population_size
        self.number_islands = number_islands

        self.initial_genome: CnnGenome = initial_genome
        
        self.islands: List[Island] = list(map(lambda _: Island(population_size), range(number_islands)))
        self.global_best_genome = initial_genome
        self.global_worst_genome = initial_genome

        self.mutation_rate: float = mutation_rate
        self.inter_island_crossover_rate: float = inter_island_crossover_rate
        self.intra_island_crossover_rate: float = intra_island_crossover_rate

        # We need to rotate through islands 0 through n - 1, this is the counter we'll use
        self.island_turn: int = 0

        # Make sure these rates sum to 1.0
        assert 0.9999 < mutation_rate + inter_island_crossover_rate + intra_island_crossover_rate < 1.00001
        

    def try_insert_genome(self, genome: CnnGenome) -> str:
        """
        Attempts to insert the supplied genome.
        If the genome is inserted, this method will return "inserted", otherwise it will return None
        If the genome is the new global best it will return "new_best"
        """

        if self.global_best_genome.fitness > genome.fitness:
            print(f"{self.global_best_genome.fitness} > {genome.fitness}")
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


    def generate_genome(self, examm: 'EXAMM'):
        self.generated_genomes += 1
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
                genome = self.islands[island_turn].get_random_genome(examm.rng)
                
            genome.island = island_turn
            
            genome = genome.copy()

            logging.info("note: we should be performing a mutation or crossover here but do not")

            return genome

