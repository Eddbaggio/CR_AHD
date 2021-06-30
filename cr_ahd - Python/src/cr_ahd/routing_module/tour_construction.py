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
                    self._carrier_insertion_construction(instance, solution, carrier,
                                                         solution.carriers[carrier].unrouted_requests)

                # when for a given request no tour can be found, create a new tour and start over
                if tour is None:
                    self._create_new_tour_with_request(instance, solution, carrier, request)

                # otherwise insert as suggested
                else:
                    self._execute_insertion(instance, solution, carrier, request, tour, pickup_pos, delivery_pos)
        pass

    @abstractmethod
    def _carrier_insertion_construction(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
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
    def _tour_cheapest_dist_insertion(instance: it.PDPInstance,
                                      solution: slt.CAHDSolution,
                                      carrier: int,
                                      tour: int,
                                      request: int):
        """Find the cheapest insertions for pickup and delivery for a given tour"""
        tour_: tr.Tour = solution.carriers[carrier].tours[tour]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

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
    For each request, identify its cheapest insertion based on distance delta. Compare the collected insertion costs
    and insert the cheapest over all requests.
    """

    def _carrier_insertion_construction(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
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
                delta, pickup_pos, delivery_pos = self._tour_cheapest_dist_insertion(instance, solution, carrier, tour,
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


class LuDessoukyPDPInsertion(PDPInsertionConstruction):
    """insertion costs are based on temporal aspects as seen in Lu,Q., & Dessouky,M.M. (2006). A new insertion-based
    construction heuristic for solving the pickup and delivery problem with time windows. European Journal of
    Operational Research, 175(2), 672–687. https://doi.org/10.1016/j.ejor.2005.05.012 """

    def _carrier_insertion_construction(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                        requests: Sequence[int]) -> Tuple[int, int, int, int]:
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier}:')
        carrier_ = solution.carriers[carrier]
        best_delta = dt.timedelta.max
        best_request = None
        best_pickup_pos = None
        best_delivery_pos = None
        best_tour = None

        for request in requests:

            best_delta_for_r = dt.timedelta.max

            for tour in range(carrier_.num_tours()):

                # cheapest way to fit request into tour based on max_shift decrease
                delta, pickup_pos, delivery_pos = self._lu_dessouky_c(instance, solution, carrier, tour, request)
                if delta < best_delta:
                    best_delta = delta
                    best_request = request
                    best_pickup_pos = pickup_pos
                    best_delivery_pos = delivery_pos
                    best_tour = tour
                if delta < best_delta_for_r:
                    best_delta_for_r = delta

            # if no feasible insertion for the current request was found, immediately return None for the tour,
            if best_delta_for_r == dt.timedelta.max:
                return request, None, None, None

        return best_request, best_tour, best_pickup_pos, best_delivery_pos

    def _lu_dessouky_c(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int,
                       request: int):
        """Find the insertions for pickup and delivery for a given tour that have the best C value"""

        tour_: tr.Tour = solution.carriers[carrier].tours[tour]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        best_delta = dt.timedelta.max
        best_pickup_position = None
        best_delivery_position = None

        for pickup_pos in range(1, len(tour_)):

            for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                # c1 + c2 + c3 = max_shift_delta; i.e. the decrease in max_shift due to the insertions
                delta = tour_.insert_max_shift_delta(instance, solution,
                                                     [pickup_pos, delivery_pos],
                                                     [pickup_vertex, delivery_vertex])

                if delta < best_delta:

                    # todo is feasibility check faster than insert_max_shift_delta computation? yes -> check
                    #  feasibility before delta!
                    if tour_.insertion_feasibility_check(instance, solution,
                                                         [pickup_pos, delivery_pos],
                                                         [pickup_vertex,
                                                          delivery_vertex]):
                        best_delta = delta
                        best_pickup_position = pickup_pos
                        best_delivery_position = delivery_pos

        return best_delta, best_pickup_position, best_delivery_position
