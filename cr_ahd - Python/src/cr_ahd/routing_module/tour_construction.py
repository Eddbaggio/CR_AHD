import datetime as dt
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, List

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

                # otherwise insert as suggested
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
                           request: int, tour: int, pickup_pos: int, delivery_pos: int):

        pickup, delivery = instance.pickup_delivery_pair(request)
        solution.carriers[carrier].tours[tour].insert_and_update(instance,
                                                                 solution,
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

        best_delta = float('inf')
        best_pickup_position = None
        best_delivery_position = None

        for pickup_pos in range(1, len(tour_)):
            for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                delta = tour_.insert_distance_delta(instance, [pickup_pos, delivery_pos],
                                                    [pickup_vertex, delivery_vertex])
                if delta < best_delta:

                    if tour_.insertion_feasibility_check(instance, solution, [pickup_pos, delivery_pos],
                                                         [pickup_vertex, delivery_vertex]):
                        best_delta = delta
                        best_pickup_position = pickup_pos
                        best_delivery_position = delivery_pos

        return best_delta, best_pickup_position, best_delivery_position

    @staticmethod
    def _create_new_tour_with_request(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        """
        In case of a multi-depot problem, the pendulum tour with the highest profit for the given request is created
        """
        carrier_ = solution.carriers[carrier]
        if carrier_.num_tours() >= instance.carriers_max_num_tours * len(solution.carrier_depots[carrier]):
            # logger.error(f'Max Vehicle Constraint violated!')
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Max. number of vehicles is {instance.carriers_max_num_tours}!'
                                              f' ({instance.id_})')

        # check all depots in case of a multi-depot instance to find the max profit pendulum tour
        max_profit = -float('inf')
        best_tour_ = False
        for depot in solution.carrier_depots[carrier]:
            tour_ = tr.Tour(carrier_.num_tours(), instance, solution, depot_index=depot)

            if tour_.insertion_feasibility_check(instance, solution, [1, 2], instance.pickup_delivery_pair(request)):
                tour_.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))

                if tour_.sum_profit > max_profit:
                    max_profit = tour_.sum_profit
                    best_tour_ = tour_

        if not best_tour_:
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Feasibility checks failed for all depots (most likely a TW problem)!')

        carrier_.tours.append(best_tour_)
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
                            routing_sequence: List[int],
                            sum_travel_distance: float,
                            sum_travel_duration: dt.timedelta,
                            arrival_schedule: List[dt.datetime],
                            service_schedule: List[dt.datetime],
                            sum_load: float,
                            sum_revenue: float,
                            sum_profit: float,
                            wait_sequence: List[dt.timedelta],
                            max_shift_sequence: List[dt.timedelta],
                            num_depots,
                            num_requests,
                            distance_matrix: Sequence[Sequence[float]],
                            vertex_load: Sequence[float],
                            revenue: Sequence[float],
                            service_duration: Sequence[dt.timedelta],
                            vehicles_max_travel_distance,
                            vehicles_max_load,
                            tw_open: Sequence[dt.datetime],
                            tw_close: Sequence[dt.datetime],
                            **kwargs
                            ):
    """
    finds the cheapest insertion position for the pickup & delivery pair for the given tour
    independent of PDPInstance class and CAHDSolution class
    """
    best_delta = float('inf')
    best_pickup_position = None
    best_delivery_position = None

    for pickup_pos in range(1, len(routing_sequence)):
        for delivery_pos in range(pickup_pos + 1, len(routing_sequence) + 1):
            delta = tr.single_insertion_distance_delta(routing_sequence=routing_sequence,
                                                       distance_matrix=distance_matrix,
                                                       insertion_indices=[pickup_pos, delivery_pos],
                                                       vertices=[pickup_vertex, delivery_vertex],
                                                       )
            if delta < best_delta:

                if tr.multi_insertion_feasibility_check(routing_sequence=routing_sequence,
                                                        sum_travel_distance=sum_travel_distance,
                                                        sum_travel_duration=sum_travel_duration,
                                                        arrival_schedule=arrival_schedule,
                                                        service_schedule=service_schedule,
                                                        sum_load=sum_load,
                                                        sum_revenue=sum_revenue,
                                                        sum_profit=sum_profit,
                                                        wait_sequence=wait_sequence,
                                                        max_shift_sequence=max_shift_sequence,
                                                        num_depots=num_depots,
                                                        num_requests=num_requests,
                                                        distance_matrix=distance_matrix,
                                                        vehicles_max_travel_distance=vehicles_max_travel_distance,
                                                        vertex_load=vertex_load,
                                                        revenue=revenue,
                                                        service_duration=service_duration,
                                                        vehicles_max_load=vehicles_max_load,
                                                        tw_open=tw_open,
                                                        tw_close=tw_close,
                                                        insertion_indices=[pickup_pos, delivery_pos],
                                                        insertion_vertices=[pickup_vertex, delivery_vertex]
                                                        ):
                    best_delta = delta
                    best_pickup_position = pickup_pos
                    best_delivery_position = delivery_pos

    return best_delta, best_pickup_position, best_delivery_position
