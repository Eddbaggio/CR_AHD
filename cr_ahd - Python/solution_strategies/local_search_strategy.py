from abc import ABC, abstractmethod

from tour import Tour
from utils import opts

"""Using the 'Strategy Pattern' to implement different routing algorithms."""


class LocalSearchStrategy(ABC):

    def __init__(self, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """
        Use a static construction method to build tours for all carriers via SEQUENTIAL CHEAPEST INSERTION.
        (Can also be used for dynamic route construction if the request-to-carrier assignment is known.)

        :param verbose:
        :param plot_level:
        :return:
        """
        self.verbose = verbose
        self.plot_level = plot_level
        self._runtime = 0

    @abstractmethod
    def optimize(self, instance):
        pass


class TwoOpt(LocalSearchStrategy):
    def optimize(self, instance):
        """
        Improve the current solution with a 2-opt local search as in
        G. A. Croes, A method for solving traveling salesman problems. Operations Res. 6 (1958)
        """
        #TODO: adding a timer to measure the performance?

        for c in instance.carriers:
            c.two_opt(instance.dist_matrix)

