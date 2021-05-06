from abc import ABC, abstractmethod
import datetime as dt
from collections import Sequence

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
            self._initialize_carrier(instance, solution, carrier)
        pass

    def _initialize_carrier(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        carrier_ = solution.carriers[carrier]
        best_request = None
        best_evaluation = -float('inf')
        for request in carrier_.unrouted_requests:
            evaluation = self._request_evaluation(instance, solution, carrier, request)
            if evaluation > best_evaluation:
                best_request = request
                best_evaluation = evaluation
        tour = tr.Tour(carrier, instance, solution, carrier)
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(best_request))
        carrier_.tours.append(tour)
        pass

    @abstractmethod
    def _request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        pass


class EarliestDueDate(TourInitializationBehavior):
    def _request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        """find request with earliest deadline"""
        _, delivery_vertex = instance.pickup_delivery_pair(request)
        tw_close = solution.tw_close(delivery_vertex)
        #  return the negative, as the procedure searched for the HIGHEST value and
        return - tw_close.total_seconds

    def __request_evaluation(self,
                             x_depot: int,
                             y_depot: int,
                             x_coords: Sequence[int],
                             y_coords: Sequence[int],
                             tw_open: Sequence,
                             tw_close: Sequence,
                             request_to_carrier_assignment: Sequence,
                             load: Sequence,
                             revenue: Sequence,
                             service_time: Sequence,
                             pickup_idx: int,
                             delivery_idx: int
                             ):
        return tw_close[delivery_idx].total_seconds


class FurthestDistance(TourInitializationBehavior):
    def _request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        midpoint_x, midpoint_y = ut.midpoint(instance, *instance.pickup_delivery_pair(request))
        depot_x = instance.x_coords[carrier]
        depot_y = instance.y_coords[carrier]
        dist = pdist([[midpoint_x, midpoint_y], [depot_x, depot_y]], 'euclidean')[0]
        return dist

    def __request_evaluation(self,
                             x_depot: int,
                             y_depot: int,
                             x_coords: Sequence[int],
                             y_coords: Sequence[int],
                             tw_open: Sequence,
                             tw_close: Sequence,
                             request_to_carrier_assignment: Sequence,
                             load: Sequence,
                             revenue: Sequence,
                             service_time: Sequence,
                             pickup_idx: int,
                             delivery_idx: int
                             ):
        x_midpoint, y_midpoint = ut.midpoint_(x_coords[pickup_idx], x_coords[delivery_idx],
                                              y_coords[pickup_idx], y_coords[delivery_idx])
        return ut.euclidean_distance(x_depot, x_midpoint, y_depot, y_midpoint)


class ClosestDistance(TourInitializationBehavior):
    def _request_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        midpoint_x, midpoint_y = ut.midpoint(instance, *instance.pickup_delivery_pair(request))
        depot_x = instance.x_coords[carrier]
        depot_y = instance.y_coords[carrier]
        dist = pdist([[midpoint_x, midpoint_y], [depot_x, depot_y]], 'euclidean')[0]
        #  return the negative, as the overall procedure searched for the HIGHEST value
        return -dist

    def __request_evaluation(self,
                             x_depot: int,
                             y_depot: int,
                             x_coords: Sequence[int],
                             y_coords: Sequence[int],
                             tw_open: Sequence,
                             tw_close: Sequence,
                             request_to_carrier_assignment: Sequence,
                             load: Sequence,
                             revenue: Sequence,
                             service_time: Sequence,
                             pickup_idx: int,
                             delivery_idx: int
                             ):
        x_midpoint, y_midpoint = ut.midpoint_(x_coords[pickup_idx], x_coords[delivery_idx],
                                              y_coords[pickup_idx], y_coords[delivery_idx])
        #  return the negative, as the overall procedure searched for the HIGHEST value
        return -ut.euclidean_distance(x_depot, x_midpoint, y_depot, y_midpoint)
