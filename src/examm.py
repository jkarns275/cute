from program_arguments import ProgramArguments
from speciation_strategy import SpeciationStrategy, IslandSpeciationStrategy
from cnn import CnnGenome


class EXAMM:

    def __init__(self, program_arguments: ProgramArguments):
        self.population_size: int = program_arguments.population_size
        self.number_islands: int = program_arguments.number_islands

        initial_genome = self.generate_initial_genome(program_arguments)

        self.speciation_strategy: SpeciationStrategy = 
            IslandSpeciationStrategy(initial_genome, self.number_islands, self.population_size)

        # Each island gets the initial genome
        self.generated_genomes = self.number_islands
        self.inserted_genomes = self.number_islands

    def unimplemented(self, m):
        raise Exception(f"method 'EXAMM::{m}' has not been implemented")

    
    def update_logs(self):
        self.unimplemented('update_logs')


    def generate_genome(self):
        genome = None
        
        # TODO: Do the generation
        self.unimplemented('generate_genome')
        
        self.generated_genomes += 1

    
    def try_insert_genome(self, genome: CnnGenome):
        self.speciation_strategy.try_insert_genome(genome)
