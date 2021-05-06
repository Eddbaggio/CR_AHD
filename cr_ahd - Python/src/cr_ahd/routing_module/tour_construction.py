import datetime as dt
from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
import logging

logger = logging.getLogger(__name__)


class TourConstructionBehavior(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def construct(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class PDPInsertionConstruction(TourConstructionBehavior):
    def construct(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            while solution.carriers[carrier].unrouted_requests:
                request, tour, pickup_pos, delivery_pos = \
                    self._carrier_cheapest_insertion(instance, solution, carrier,
                                                     solution.carriers[carrier].unrouted_requests)
                # when for a given request no tour can be found, create a new tour and start over
                if tour is None:
                    self._create_new_tour_with_request(instance, solution, carrier, request)
                else:
                    self._execute_insertion(instance, solution, carrier, request, tour, pickup_pos, delivery_pos)
        pass

    @abstractmethod
    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                                    requests: Sequence[int]) -> Tuple[int, int, int, int]:
        """

        :return: tuple of (request, tour, pickup_pos, delivery_pos) for the insertion operation
        """

        pass

    @staticmethod
    def _execute_insertion(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                           request, tour, pickup_pos, delivery_pos):

        pickup, delivery = instance.pickup_delivery_pair(request)
        solution.carriers[carrier].tours[tour].insert_and_update(instance, solution,
                                                                 [pickup_pos, delivery_pos],
                                                                 [pickup, delivery])
        solution.carriers[carrier].unrouted_requests.remove(request)

    @staticmethod
    def _tour_cheapest_insertion(instance: it.PDPInstance,
                                 solution: slt.GlobalSolution,
                                 carrier: int, tour: int,
                                 unrouted_request: int):
        """Find the cheapest insertions for pickup and delivery for a given tour"""
        tour_: tr.Tour = solution.carriers[carrier].tours[tour]

        best_delta = float('inf')
        best_pickup_position = None
        best_delivery_position = None

        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(unrouted_request)

        for pickup_pos in range(1, len(tour_)):
            for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):  # TODO I only have to check those positions that would not violate the TWs of the other nodes, thus I can potentially start even later that i+1
                delta = tour_.insertion_distance_delta(instance, [pickup_pos, delivery_pos],
                                                       [pickup_vertex, delivery_vertex])
                if delta < best_delta:
                    if tour_.insertion_feasibility_check(instance, solution, [pickup_pos, delivery_pos],
                                                         [pickup_vertex, delivery_vertex]):
                        best_delta = delta
                        best_pickup_position = pickup_pos
                        best_delivery_position = delivery_pos
        return best_delta, best_pickup_position, best_delivery_position

    @staticmethod
    def _create_new_tour_with_request(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                                      request: int):
        carrier_ = solution.carriers[carrier]
        if carrier_.num_tours() >= instance.carriers_max_num_tours:
            # logger.error(f'Max Vehicle Constraint violated!')
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Max. number of vehicles is {instance.carriers_max_num_tours}!')
        rtmp = tr.Tour(len(carrier_.tours), instance, solution, carrier_.id_)
        rtmp.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        carrier_.tours.append(rtmp)
        carrier_.unrouted_requests.remove(request)
        return


class CheapestPDPInsertion(PDPInsertionConstruction):
    """
    For each REQUEST, identify its cheapest insertion. Compare the collected insertion costs and insert the cheapest
    over all requests.
    """

    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                                    requests:Sequence[int]):
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier}:')
        carrier_ = solution.carriers[carrier]
        best_delta = float('inf')
        best_request = best_pickup_pos = best_delivery_pos = best_tour = None
        for request in requests:
            best_delta_for_r = float('inf')
            for tour in range(carrier_.num_tours()):
                # cheapest way to fit request into tour
                delta, pickup_pos, delivery_pos = self._tour_cheapest_insertion(instance, solution, carrier, tour,
                                                                                request)
                if delta < best_delta:
                    best_delta = delta
                    best_request = request
                    best_pickup_pos = pickup_pos
                    best_delivery_pos = delivery_pos
                    best_tour = tour
                if delta < best_delta_for_r:
                    best_delta_for_r = delta
            # if no feasible insertion for the current request was found, return None for the tour
            if best_delta_for_r == float('inf'):
                return request, None, None, None
        return best_request, best_tour, best_pickup_pos, best_delivery_pos

def cheapest_insertion_tour(routing_sequence: Sequence[int],
                            x_coords: Sequence[int],
                            y_coords: Sequence[int],
                            pickup: int,
                            delivery:int,
                            # TODO ... create a cheapest insertion on the most abstract level using only basic data types
                            ):
