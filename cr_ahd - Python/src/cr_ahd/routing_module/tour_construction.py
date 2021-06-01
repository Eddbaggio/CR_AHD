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
    def construct(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        pass


class PDPInsertionConstruction(TourConstructionBehavior, ABC):
    def construct(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
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
    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                    requests: Sequence[int]) -> Tuple[int, int, int, int]:
        """

        :return: tuple of (request, tour, pickup_pos, delivery_pos) for the insertion operation
        """

        pass

    @staticmethod
    def _execute_insertion(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                           request, tour, pickup_pos, delivery_pos):

        pickup, delivery = instance.pickup_delivery_pair(request)
        solution.carriers[carrier].tours[tour].insert_and_update(instance, solution,
                                                                 [pickup_pos, delivery_pos],
                                                                 [pickup, delivery])
        solution.carriers[carrier].unrouted_requests.remove(request)

    @staticmethod
    def _tour_cheapest_insertion(instance: it.PDPInstance,
                                 solution: slt.CAHDSolution,
                                 carrier: int, tour: int,
                                 unrouted_request: int):
        """Find the cheapest insertions for pickup and delivery for a given tour"""
        tour_: tr.Tour = solution.carriers[carrier].tours[tour]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(unrouted_request)

        return tour_cheapest_insertion(
            pickup_vertex=pickup_vertex,
            delivery_vertex=delivery_vertex,
            routing_sequence=tour_.routing_sequence,
            travel_distance_sequence=tour_.travel_distance_sequence,
            service_schedule=tour_.service_schedule,
            load_sequence=tour_.load_sequence,
            num_carriers=instance.num_carriers,
            num_requests=instance.num_requests,
            distance_matrix=instance._distance_matrix,
            vehicles_max_travel_distance=instance.vehicles_max_travel_distance,
            vertex_load=instance.load,
            service_duration=instance.service_duration,
            vehicles_max_load=instance.vehicles_max_load,
            tw_open=solution.tw_open,
            tw_close=solution.tw_close,
        )

    @staticmethod
    def _create_new_tour_with_request(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                      request: int):
        carrier_ = solution.carriers[carrier]
        if carrier_.num_tours() >= instance.carriers_max_num_tours:
            # logger.error(f'Max Vehicle Constraint violated!')
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Max. number of vehicles is {instance.carriers_max_num_tours}!')
        rtmp = tr.Tour(carrier_.num_tours(), instance, solution, carrier_.id_)
        rtmp.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        carrier_.tours.append(rtmp)
        carrier_.unrouted_requests.remove(request)
        return


class CheapestPDPInsertion(PDPInsertionConstruction):
    """
    For each request, identify its cheapest insertion. Compare the collected insertion costs and insert the cheapest
    over all requests.
    """

    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                    requests: Sequence[int]):
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier}:')
        carrier_ = solution.carriers[carrier]
        best_delta = float('inf')
        best_request = None
        best_pickup_pos = None
        best_delivery_pos = None
        best_tour = None
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


# =====================================================================================================================
# functions
# =====================================================================================================================


def tour_cheapest_insertion(pickup_vertex: int,
                            delivery_vertex: int,
                            routing_sequence: Sequence[int],
                            travel_distance_sequence: Sequence[float],
                            service_schedule: Sequence[dt.datetime],
                            load_sequence: Sequence[float],
                            num_carriers: int,
                            num_requests: int,
                            distance_matrix: Sequence[Sequence[float]],
                            vehicles_max_travel_distance: float,
                            vertex_load: Sequence[int],
                            service_duration: Sequence[dt.timedelta],
                            vehicles_max_load: float,
                            tw_open: Sequence[dt.datetime],
                            tw_close: Sequence[dt.datetime],
                            **kwargs
                            ):
    """
    independent of PDPInstance class and GlobalSolution class
    """
    best_delta = float('inf')
    best_pickup_position = None
    best_delivery_position = None

    for pickup_pos in range(1, len(routing_sequence)):
        # TODO I only have to check those positions that would not violate the TWs of the other nodes, thus I can
        #  potentially start even later than i+1. There are functions somewhere that do this, afaik
        for delivery_pos in range(pickup_pos + 1, len(routing_sequence) + 1):
            delta = tr.insertion_distance_delta(
                routing_sequence=routing_sequence,
                distance_matrix=distance_matrix,
                insertion_indices=[pickup_pos, delivery_pos],
                vertices=[pickup_vertex, delivery_vertex],
            )
            if delta < best_delta:
                if tr.insertion_feasibility_check(routing_sequence=routing_sequence,
                                                  travel_distance_sequence=travel_distance_sequence,
                                                  service_schedule=service_schedule,
                                                  load_sequence=load_sequence,
                                                  num_carriers=num_carriers,
                                                  num_requests=num_requests,
                                                  distance_matrix=distance_matrix,
                                                  vehicles_max_travel_distance=vehicles_max_travel_distance,
                                                  vertex_load=vertex_load,
                                                  service_duration=service_duration,
                                                  vehicles_max_load=vehicles_max_load,
                                                  tw_open=tw_open,
                                                  tw_close=tw_close,
                                                  insertion_positions=[pickup_pos, delivery_pos],
                                                  insertion_vertices=[pickup_vertex, delivery_vertex]):
                    best_delta = delta
                    best_pickup_position = pickup_pos
                    best_delivery_position = delivery_pos
    return best_delta, best_pickup_position, best_delivery_position
