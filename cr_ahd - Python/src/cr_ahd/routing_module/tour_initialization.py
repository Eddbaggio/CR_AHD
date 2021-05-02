from abc import ABC, abstractmethod
import datetime as dt

from scipy.spatial.distance import pdist

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut


class TourInitializationBehavior(ABC):
    """
    Visitor Interface to apply a tour initialization heuristic to either an instance (i.e. each of its carriers)
    or a single specific carrier. Contrary to the routing visitor, this one will only allocate a single seed request
    """

    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            self.initialize_carrier(instance, solution, carrier)
        pass

    def initialize_carrier(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        carrier_ = solution.carriers[carrier]
        best_request = None
        best_evaluation = -float('inf')
        for request in carrier_.unrouted_requests:
            evaluation = self.request_evaluation(instance, solution, carrier, request)
            if evaluation > best_evaluation:
                best_request = request
                best_evaluation = evaluation
        tour = tr.Tour(carrier, instance, solution, carrier)
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(best_request))
        carrier_.tours.append(tour)
        pass

    @abstractmethod
    def request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        pass


class EarliestDueDate(TourInitializationBehavior):
    def request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        """find request with earliest deadline"""
        _, delivery_vertex = instance.pickup_delivery_pair(request)
        tw_close = solution.tw_close(delivery_vertex)
        #  return the negative, as the procedure searched for the HIGHEST value and
        return - tw_close.total_seconds


class FurthestDistance(TourInitializationBehavior):
    def request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        midpoint_x, midpoint_y = ut.midpoint(instance, *instance.pickup_delivery_pair(request))
        depot_x = instance.x_coords[carrier]
        depot_y = instance.y_coords[carrier]
        dist = pdist([[midpoint_x, midpoint_y], [depot_x, depot_y]], 'euclidean')[0]
        return dist


class ClosestDistance(TourInitializationBehavior):

    def request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        midpoint_x, midpoint_y = ut.midpoint(instance, *instance.pickup_delivery_pair(request))
        depot_x = instance.x_coords[carrier]
        depot_y = instance.y_coords[carrier]
        dist = pdist([[midpoint_x, midpoint_y], [depot_x, depot_y]], 'euclidean')[0]
        #  return the negative, as the procedure searched for the HIGHEST value and
        return -dist
