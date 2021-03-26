from abc import ABC, abstractmethod

from src.cr_ahd.utility_module.utils import opts, InsertionError


class TourImprovementBehavior(ABC):
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
    def improve_instance(self, instance):
        pass

    @abstractmethod
    def improve_carrier(self, carrier):
        pass

    @abstractmethod
    def improve_tour(self, tour):
        pass


class TwoOpt(TourImprovementBehavior):
    """
    Improve the current solution with a 2-opt local search as in
    G. A. Croes, A method for solving_module traveling salesman problems. Operations Res. 6 (1958)
    """

    def improve_instance(self, instance):
        for carrier in instance.carriers:
            self.improve_carrier(carrier)
        # instance.finalizing_visitor = self
        # instance._finalized = True
        pass

    def improve_carrier(self, carrier):
        """Applies the 2-Opt local search operator to all vehicles/tours"""
        if self.verbose > 0:
            print(f'2-opt finalizing for {carrier}')
        for vehicle in carrier.active_vehicles:
            self.improve_tour(vehicle.tour)
        # carrier._finalized = True
        pass

    def improve_tour(self, tour):
        best_cost = tour.sum_travel_durations
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
                        if tour.sum_travel_durations < best_cost:
                            improved = True
                            best_cost = tour.sum_travel_durations
                        # if no improvement -> undo the 2opt swap
                        else:
                            tour.reverse_section(i, j)
                    except InsertionError as e:
                        continue  # if reversal is infeasible, continue with next iteration

        # tour._finalized = True
        pass


# class ThreeOpt(TourImprovementBehavior):
#     pass

# class LinKernighan(TourImprovementBehavior):
#     pass

# class Swap(TourImprovementBehavior):
#     pass

class NoImprovement(TourImprovementBehavior):
    def improve_instance(self, instance):
        pass

    def improve_carrier(self, carrier):
        pass

    def improve_tour(self, tour):
        pass
