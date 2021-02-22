from abc import ABC, abstractmethod

from helper.utils import opts, InsertionError


class InitializingVisitor(ABC):
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

    # a method initialize_tour (or initialize_vehicle) is not required as the initialization on a tour-level does not
    # depend on the visitor, i.e. on the initialization strategy: the tour cannot distinguish between earliest due
    # date or furthest distance as it does not have access to the vertex information that shall be inserted

    @abstractmethod
    def find_seed_request(self, carrier):
        pass


class EarliestDueDate(InitializingVisitor):
    """
    Visitor for building a pendulum tour for
    a) every carrier's first inactive vehicle
    b) a unique carrier's first inactive vehicle
    by finding the vertex with earliest due date
    """

    def find_seed_request(self, carrier):
        """find request with earliest deadline"""
        seed = carrier.unrouted[0]
        for request in carrier.unrouted:
            if request.tw.l < seed.tw.l:
                seed = request
        return seed

    def initialize_instance(self, instance):
        for carrier in instance.carriers:
            carrier.initialize(EarliestDueDate(self.verbose, self.plot_level))
        instance._initialized = True
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
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        carrier._initialized = True
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
"""
