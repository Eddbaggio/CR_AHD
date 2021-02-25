from abc import ABC, abstractmethod
import instance as it
import carrier as cr


class PendulumInitializer(ABC):
    @abstractmethod
    def find_seed(self, carrier: cr.Carrier):
        pass

    @abstractmethod
    def initialize_instance(self, instance: it.Instance):
        pass

    @abstractmethod
    def initialize_carrier(self, carrier: cr.Carrier):
        pass


class EarliestDueDate(PendulumInitializer):
    """
    Visitor for building a pendulum tour for
    a) every carrier's first inactive vehicle
    b) a unique carrier's first inactive vehicle
    by finding the vertex with earliest due date
    """

    def find_seed(self, carrier):
        """find request with earliest deadline"""
        seed = carrier.unrouted_requests[0]
        for request in carrier.unrouted_requests:
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
        if len(carrier.unrouted_requests) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed(carrier)
            vehicle.tour.insert_and_update(index=1, vertex=seed)
            if vehicle.tour.is_feasible():
                vehicle.tour.compute_cost_and_schedules()
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        carrier._initialized = True
        pass