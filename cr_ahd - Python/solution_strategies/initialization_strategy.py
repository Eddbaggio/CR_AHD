from abc import ABC, abstractmethod
from utils import opts, InsertionError


class RouteInitializationStrategy(ABC):

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
        # TODO: adding a timer to measure the performance?

    def initialize(self, instance):
        pass

    def initialize(self, carrier):
        pass

    @abstractmethod
    def find_seed_request(self, carrier):
        pass


class EarliestDueDate_Instance(RouteInitializationStrategy):
    def initialize(self, instance):
        """
        Builds a pendulum tour for every carrier's first inactive vehicle by finding a the vertex with earliest due date
        """
        for c in instance.carriers:
            vehicle = c.inactive_vehicles[0]
            assert len(vehicle.tour) == 2, 'Vehicle already has a tour'
            if len(c.unrouted) > 0:
                # find request with earliest deadline and initialize pendulum tour
                seed = self.find_seed_request(c)
                vehicle.tour.insert_and_reset(index=1, vertex=seed)
                if vehicle.tour.is_feasible(dist_matrix=instance.dist_matrix):
                    vehicle.tour.compute_cost_and_schedules(dist_matrix=instance.dist_matrix)
                    c.unrouted.pop(seed.id_)
                else:
                    raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
            return

    def find_seed_request(self, carrier):
        """find request with earliest deadline"""
        seed = list(carrier.unrouted.values())[0]
        for key, request in carrier.unrouted.items():
            if request.tw.l < seed.tw.l:
                seed = carrier.unrouted[key]
        return seed

class EarliestDueDate_Carrier(RouteInitializationStrategy):
    def initialize(self, carrier):


class FurthestDistance(RouteInitializationStrategy):
    def initialize(self, instance):
        raise NotImplementedError
        pass

    def find_seed_request(self, carrier):
        """find request with furthest distance from carriers depot"""
        raise NotImplementedError
        pass
