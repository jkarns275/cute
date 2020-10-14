import sys
import logging
from pathlib import Path

import hp

if False:
    from cute import Cute

class HPLog:

    def __init__(self, directory: str):
        self.path_str: str = f"{directory}/"
        self.path: Path = Path(self.path_str)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path_str += "hp_log.csv"
        try:
            with open(self.path_str, "w+") as f:
                # Creates and cleares the file if it already exists
                pass
        except Exception as e:
            logging.fatal("Encountered fatal error when trying to open hyperparameter log: \n" + str(e))
            sys.exit(-1)

        self.write_header()

    def try_write(self, line):
        if line[-1] != "\n":
            line = line + "\n"

        try:
            with open(self.path_str, "a+") as f:
                f.write(line)
                f.flush()
        except Exception as e:
            logging.fatal("Encountered fatal error when trying to write to hyperparameter log: \n" + str(e))
            sys.exit(-1)

    def write_header(self):
        line = "Genomes Generated" 
        for parameter in hp.EvolvableHPConfig.PARAMETERS.keys():
            line += f", {parameter}"

        self.try_write(line)
    
    def update_log(self, cute: 'Cute'):
        best_genome = cute.get_best_genome()
        hps = best_genome.hp
        line = f"{cute.get_generated_genomes()}"
        for parameter in hp.EvolvableHPConfig.PARAMETERS.keys():
            value = getattr(hps, parameter)
            line += f", {value}"

        self.try_write(line)
