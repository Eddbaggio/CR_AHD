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
            evaluation = self._request_evaluation(*instance.pickup_delivery_pair(request),
                                                  **{'x_depot': instance.x_coords[carrier],
                                                     'y_depot': instance.y_coords[carrier],
                                                     'x_coords': instance.x_coords,
                                                     'y_coords': instance.y_coords,
                                                     })
            if evaluation > best_evaluation:
                best_request = request
                best_evaluation = evaluation
        tour = tr.Tour(carrier, instance, solution, carrier)
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(best_request))
        carrier_.tours.append(tour)
        pass

    @abstractmethod
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs):
        r"""
        :param pickup_idx:
        :param delivery_idx:
        :param kwargs: \**kwargs:
        See below

        :Keyword Arguments:
        * *x_depot*  --
        * *y_depot*  --
        * *x_coords*  --
        * *y_coords*  --
        * *tw_open*  --
        * *tw_close*  --

        :return:
        """
        pass


class EarliestDueDate(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        return - kwargs['tw_close'][delivery_idx].total_seconds


class FurthestDistance(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        x_midpoint, y_midpoint = ut.midpoint_(kwargs['x_coords'][pickup_idx], kwargs['x_coords'][delivery_idx],
                                              kwargs['y_coords'][pickup_idx], kwargs['y_coords'][delivery_idx])
        return ut.euclidean_distance(kwargs['x_depot'], kwargs['y_depot'], x_midpoint, y_midpoint)


class ClosestDistance(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        x_midpoint, y_midpoint = ut.midpoint_(kwargs['x_coords'][pickup_idx], kwargs['x_coords'][delivery_idx],
                                              kwargs['y_coords'][pickup_idx], kwargs['y_coords'][delivery_idx])
        return - ut.euclidean_distance(kwargs['x_depot'], kwargs['y_depot'], x_midpoint, y_midpoint)
