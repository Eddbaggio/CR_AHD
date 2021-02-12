from abc import ABC, abstractmethod

from utils import opts, InsertionError


class InitializationVisitor(ABC):
    """
    Visitor Interface to apply a tour initialization heuristic to either an instance (i.e. each of its carriers)
    or a single specific carrier. Contrary to the routing visitor, this one will only allocate a single seed request
    """

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
    def initialize_instance(self, instance):
        pass

    @abstractmethod
    def initialize_carrier(self, carrier):
        pass

    @abstractmethod
    def initialize_vehicle(self, vehicle):
        pass

    @abstractmethod
    def find_seed_request(self, carrier):
        pass


class EarliestDueDate(InitializationVisitor):
    """
    Visitor for building a pendulum tour for
    a) every carrier's first inactive vehicle
    b) a unique carrier's first inactive vehicle
    c) a unique inactive vehicle
    by finding the vertex with earliest due date
    """

    def find_seed_request(self, carrier):
        """find request with earliest deadline"""
        seed = list(carrier.unrouted.values())[0]
        for key, request in carrier.unrouted.items():
            if request.tw.l < seed.tw.l:
                seed = carrier.unrouted[key]
        return seed

    def initialize_instance(self, instance):
        for c in instance.carriers:
            c.initialize(EarliestDueDate(self.verbose, self.plot_level))
        return

    def initialize_carrier(self, carrier):
        vehicle = carrier.inactive_vehicles[0]
        assert len(vehicle.tour) == 2, 'Vehicle already has a tour'
        if len(carrier.unrouted) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed_request(carrier)
            vehicle.tour.insert_and_reset(index=1, vertex=seed)
            if vehicle.tour.is_feasible():
                vehicle.tour.compute_cost_and_schedules()
                carrier.unrouted.pop(seed.id_)
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        pass

    def initialize_vehicle(self, vehicle):
        # will be tricky because we need access to the carrier's requests. But maybe this one won't even be necessary
        # in the end
        pass


"""
class FurthestDistance(InitializationVisitor):

    def find_seed_request(self, carrier):
        raise NotImplementedError
        pass

    def initialize_instance(self, instance):
        raise NotImplementedError
        pass

    def initialize_carrier(self, carrier):
        raise NotImplementedError
        pass

    def initialize_vehicle(self, vehicle):
        raise NotImplementedError
        pass
"""
