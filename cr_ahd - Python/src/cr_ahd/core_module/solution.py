from pathlib import Path
import json


class Solution(object):
    def __init__(self):
        self.carriers = None
        # Todo? I do not need the complete carrier information here!
        #  remove Tour class from instance and only store tours in a solution!?
        #  decoupling: an instance has all the required input data, the solution has all the produced output data
        self.solution_algorithm = None

    def write_to_json(self):
        pass

    def plot(self):
        pass


def read_solution_and_summary_from_json(path: Path):
    with open(path, mode='r') as f:
        solution, summary = json.load(f).values()
    return solution, summary
