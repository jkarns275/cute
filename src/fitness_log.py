import sys
import logging
from pathlib import Path

if False:
    from cute import Cute


class FitnessLog:

    LOG_ITEMS = (
        "Genomes Generated",
        "Genomes Inserted",
        "Best Fitness",
        "Best Accuracy",
        "Worst Fitness",
        "Worst Accuracy"
    )


    LOG_HANDLERS = {
        "Genomes Generated":    "get_genomes_generated",
        "Genomes Inserted":     "get_genomes_inserted",
        "Best Fitness":         "get_best_fitness",
        "Best Accuracy":        "get_best_accuracy",
        "Worst Fitness":        "get_worst_fitness",
        "Worst Accuracy":       "get_worst_accuracy",
    }

    
    def get_genomes_generated(self, cute: 'Cute') -> int:
        return cute.get_generated_genomes()


    def get_genomes_inserted(self, cute: 'Cute') -> int:
        return cute.get_inserted_genomes()


    def get_best_fitness(self, cute: 'Cute') -> float:
        return cute.get_best_fitness()


    def get_worst_fitness(self, cute: 'Cute') -> float:
        return cute.get_worst_fitness()


    def get_best_accuracy(self, cute: 'Cute') -> float:
        return cute.get_best_accuracy()


    def get_worst_accuracy(self, cute: 'Cute') -> float:
        return cute.get_worst_accuracy()


    def __init__(self, directory: str):
        self.path_str: str = f"{directory}/"
        self.path: Path = Path(self.path_str)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path_str += "fitness_log.csv"

        try:
            with open(self.path_str, "w+") as f:
                # Creates and cleares the file if it already exists
                pass
        except Exception as e:
            logging.fatal("Encountered fatal error when trying to open fitness log: \n" + str(e))
            sys.exit(-1)

        self.write_header()


    def write_header(self):
        header = ", ".join(FitnessLog.LOG_ITEMS)

        self.try_write(header)

    
    def try_write(self, line):
        if line[-1] != "\n":
            line = line + "\n"

        try:
            with open(self.path_str, "a+") as f:
                f.write(line)
                f.flush()
        except Exception as e:
            logging.fatal("Encountered fatal error when trying to write to fitness log: \n" + str(e))
            sys.exit(-1)


    def close(self):
        pass


    def update_log(self, cute: 'Cute'):
        line_items =[]

        for item in FitnessLog.LOG_ITEMS:
            handler_name = FitnessLog.LOG_HANDLERS[item]
            assert hasattr(self, handler_name)
            handler = getattr(self, handler_name)
            line_items.append(str(handler(cute)))

        line = ", ".join(line_items)

        self.try_write(line)
