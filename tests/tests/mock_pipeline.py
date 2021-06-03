import logging
import math


class Pipeline:
    def __init__(self, conf):
        logging.info(f'Instantiating pipeline with {conf}')
        self.config = conf

    def __call__(self, data, steps):
        for step in steps:
            logging.info(f"Performing step {step}")
            math.factorial(50000)
        return data + "-".join(steps)
