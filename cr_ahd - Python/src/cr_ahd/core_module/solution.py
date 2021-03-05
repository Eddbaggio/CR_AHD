class Solution(object):
    def __init__(self):
        self.carriers = None
        # Todo? I do not need the complete carrier information here!
        #  remove Tour class from instance and only store tours in a solution!?
        #  decoupling: an instance has all the required input data, the solution has all the produced output data
        self.solution_algorithm = None

    def write_to_json(self):
        pass

    def read_from_json(self):
        pass

    def plot(self):
        pass

    def metrics(self):
        """dict of all the features of the solution, incl. total cost, number of active vehicles, ..."""
        pass
