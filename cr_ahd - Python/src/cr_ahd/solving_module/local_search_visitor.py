from abc import ABC, abstractmethod

from src.cr_ahd.utility_module.utils import opts, InsertionError


class FinalizingVisitor(ABC):
    """Visitor Interface to apply a FINAL local search heuristic to either an instance, a carrier or a single tour"""

    def __init__(self, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """

        :param verbose:
        :param plot_level:
        """
        self.verbose = verbose
        self.plot_level = plot_level
        self._runtime = 0
        # TODO: adding a timer to measure the performance?

    @abstractmethod
    def finalize_instance(self, instance):
        pass

    @abstractmethod
    def finalize_carrier(self, carrier):
        pass

    @abstractmethod
    def finalize_tour(self, tour):
        pass


class TwoOpt(FinalizingVisitor):
    """
    Improve the current solution with a 2-opt local search as in
    G. A. Croes, A method for solving_module traveling salesman problems. Operations Res. 6 (1958)
    """

    def finalize_instance(self, instance):
        for carrier in instance.carriers:
            self.finalize_carrier(carrier)
        # instance.finalizing_visitor = self
        # instance._finalized = True
        pass

    def finalize_carrier(self, carrier):
        """Applies the 2-Opt local search operator to all vehicles/tours"""
        if self.verbose > 0:
            print(f'2-opt finalizing for {carrier}')
        for vehicle in carrier.active_vehicles:
            self.finalize_tour(vehicle.tour)
        # carrier._finalized = True
        pass

    def finalize_tour(self, tour):
        best_cost = tour.cost
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue  # no effect
                    try:
                        # the actual 2opt swap
                        tour.reverse_section(i, j)
                        if tour.cost < best_cost:
                            improved = True
                            best_cost = tour.cost
                        # if no improvement -> undo the 2opt swap
                        else:
                            tour.reverse_section(i, j)
                    except InsertionError as e:
                        continue  # if reversal is infeasible, continue with next iteration

        # tour._finalized = True
        pass


# class Swap(FinalizationVisitor):
#     pass

class NoLocalSearch(FinalizingVisitor):
    def finalize_instance(self, instance):
        pass

    def finalize_carrier(self, carrier):
        pass

    def finalize_tour(self, tour):
        pass
