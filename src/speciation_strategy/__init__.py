if False:
    from cute import Cute

class SpeciationStrategy:

    def __init__(self):
        self.generated_genomes: int = 0
        self.inserted_genomes: int = 0


    def try_insert_genome(self, genome):
        """
        Attempts to insert the supplied genome.
        If the genome is inserted, this method will return True, otherwise it will return False.
        """
        raise Exception("called abstract insert_genome method")


    def get_best_fitness(self):
        raise Exception("called abstract get_best_fitness")


    def get_worst_fitness(self):
        raise Exception("called abstact get_worst_fitness")


    def get_generated_genomes(self):
        raise Exception("called abstract get_generated_genomes")


    def get_inserted_genomes(self):
        raise Exception("called abstract get_inserted_genomes")


    def get_best_genome(self):
        raise Exception("called abstract get_best_genome")


    def get_genome_for_mutation(self, cute: 'Cute'):
        raise Exception("called abstract get_genome_for_mutation")
