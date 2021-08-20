import datetime as dt
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

from src.cr_ahd.utility_module import utils as ut, profiling as pr
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr

logger = logging.getLogger(__name__)


class PDPParallelInsertionConstruction(ABC):
    def construct_static(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        """inserts ALL unrouted requests of the specified carrier"""
        carrier_ = solution.carriers[carrier]
        while carrier_.unrouted_requests:
            request, tour, pickup_pos, delivery_pos = self.best_insertion_for_carrier(instance, solution, carrier)

            # when for a given request no tour can be found, create a new tour and start over
            # this will fail if max_num_vehicles is exceeded
            if tour is None:
                self.create_new_tour_with_request(instance, solution, carrier, request)

            # otherwise insert as suggested
            else:
                self.execute_insertion(instance, solution, carrier, request, tour, pickup_pos, delivery_pos)

        pass

    def construct_dynamic(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        """take the single (!) unrouted customer of the given carrier and insert it in the best position
        alters the solution IN PLACE"""
        carrier_ = solution.carriers[carrier]
        request = carrier_.unrouted_requests[0]
        insertion_criteria, tour, pickup_pos, delivery_pos = self.best_insertion_for_request(instance, solution,
                                                                                             carrier, request)
        if tour is None:
            # print(f'{request}: new tour')
            self.create_new_tour_with_request(instance, solution, carrier, request)
        else:
            # print(f'{request}: t{carrier_.tours[tour].routing_sequence[0]} {pickup_pos, delivery_pos}')
            self.execute_insertion(instance, solution, carrier, request, tour, pickup_pos, delivery_pos)

        pass

    def best_insertion_for_carrier(
            self,
            instance: it.PDPInstance,
            solution: slt.CAHDSolution,
            carrier: int) -> Tuple[int, Union[None, int], Union[None, int], Union[None, int]]:
        """
        Scanning through all the unrouted requests of the given carrier, the best one is identified and returned as
        a tuple of (request, tour, pickup_pos, delivery_pos). "Best" in this case is defined by the inheriting class
        (e.g. lowest distance increase or smallest time shift)

        :return: the best found insertion as a tuple of (request, tour, pickup_pos, delivery_pos)
        """
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier}:')
        carrier_ = solution.carriers[carrier]

        best_delta = float('inf')
        best_request = None
        best_tour = None
        best_pickup_pos = None
        best_delivery_pos = None

        for request in carrier_.unrouted_requests:

            delta, tour, pickup_pos, delivery_pos = self.best_insertion_for_request(instance, solution, carrier,
                                                                                    request)

            if delta < best_delta:
                best_delta = delta
                best_request = request
                best_tour = tour
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

            # if no feasible insertion for the current request was found, return None for the tour
            if best_delta == float('inf'):
                return request, None, None, None

        return best_request, best_tour, best_pickup_pos, best_delivery_pos

    def best_insertion_for_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                   request: int):
        """For the given request, finds the best combination of (a) tour, (b) pickup position and (c) delivery position
         for the best insertion. Best, in this case, is defined by the inheriting class (e.g. lowest cost increase or
         least time shift).

         :returns: delta, tour, pickup_position, delivery_position of the best found insertion
         """
        best_delta = float('inf')
        best_tour = None
        best_pickup_pos = None
        best_delivery_pos = None

        for tour in range(solution.carriers[carrier].num_tours()):

            delta, pickup_pos, delivery_pos = self.best_insertion_for_request_in_tour(instance, solution,
                                                                                      solution.carriers[carrier].tours[
                                                                                          tour], request)
            if delta < best_delta:
                best_delta = delta
                best_tour = tour
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

        return best_delta, best_tour, best_pickup_pos, best_delivery_pos

    @abstractmethod
    def best_insertion_for_request_in_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_,
                                           request: int, check_feasibility=True) -> Tuple[float, int, int]:
        """returns the feasible insertion move with the lowest cost as a tuple of (delta, pickup_position,
        delivery_position)
        :param check_feasibility:
        """
        pass

    @staticmethod
    def execute_insertion(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                          request: int, tour: int, pickup_pos: int, delivery_pos: int):

        pickup, delivery = instance.pickup_delivery_pair(request)
        tour_ = solution.carriers[carrier].tours[tour]
        tour_.insert_and_update(instance, solution, [pickup_pos, delivery_pos], [pickup, delivery])
        solution.request_to_tour_assignment[instance.request_from_vertex(pickup)] = tour
        solution.carriers[carrier].unrouted_requests.remove(request)
        solution.carriers[carrier].routed_requests.append(request)

    @staticmethod
    def execute_insertion_in_tour(instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour,
                                  request: int, pickup_pos: int, delivery_pos: int):

        pickup, delivery = instance.pickup_delivery_pair(request)
        tour_.insert_and_update(instance, solution, [pickup_pos, delivery_pos], [pickup, delivery])
        solution.request_to_tour_assignment[instance.request_from_vertex(pickup)] = tour_.id_

    @staticmethod
    def create_new_tour_with_request(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
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
                solution.request_to_tour_assignment[request] = tour_.id_

                if tour_.sum_profit > max_profit:
                    max_profit = tour_.sum_profit
                    best_tour_ = tour_

        if not best_tour_:
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Feasibility checks failed for all depots (most likely a TW problem)!')

        carrier_.tours.append(best_tour_)
        carrier_.unrouted_requests.remove(request)
        carrier_.routed_requests.append(request)
        return


class MinTravelDistanceInsertion(PDPParallelInsertionConstruction):
    """
    For each request, identify its cheapest insertion based on distance delta. Compare the collected insertion costs
    and insert the cheapest over all requests.
    """

    @pr.timing
    def best_insertion_for_request_in_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_,
                                           request: int, check_feasibility=True) -> Tuple[float, int, int]:
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        best_delta = float('inf')
        best_pickup_position = None
        best_delivery_position = None

        for pickup_pos in range(1, len(tour_)):
            for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                delta = tour_.insert_distance_delta(instance, [pickup_pos, delivery_pos],
                                                    [pickup_vertex, delivery_vertex])
                if delta < best_delta:

                    update_best = True
                    if check_feasibility:
                        update_best = tour_.insertion_feasibility_check(instance, solution, [pickup_pos, delivery_pos],
                                                                        [pickup_vertex, delivery_vertex])
                    if update_best:
                        best_delta = delta
                        best_pickup_position = pickup_pos
                        best_delivery_position = delivery_pos

        return best_delta, best_pickup_position, best_delivery_position


class MinTimeShiftInsertion(PDPParallelInsertionConstruction):
    """insertion costs are based on temporal aspects as seen in Lu,Q., & Dessouky,M.M. (2006). A new insertion-based
    construction heuristic for solving the pickup and delivery problem with time windows. European Journal of
    Operational Research, 175(2), 672–687. https://doi.org/10.1016/j.ejor.2005.05.012 """

    @pr.timing
    def best_insertion_for_request_in_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_,
                                           request: int, check_feasibility=True) -> Tuple[float, int, int]:
        """Find the insertions for pickup and delivery for a given tour that have the best C value
        :param check_feasibility:
        :param tour_:
        """

        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        best_delta = dt.timedelta.max
        best_pickup_pos = None
        best_delivery_pos = None

        for pickup_pos in range(1, len(tour_)):

            for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                # c1 + c2 + c3 = max_shift_delta; i.e. the decrease in max_shift due to the insertions
                delta = tour_.insert_max_shift_delta(instance, solution,
                                                     [pickup_pos, delivery_pos],
                                                     [pickup_vertex, delivery_vertex])

                update_best = True
                if check_feasibility:
                    update_best = tour_.insertion_feasibility_check(instance, solution, [pickup_pos, delivery_pos],
                                                                    [pickup_vertex, delivery_vertex])
                if update_best:
                    best_delta = delta
                    best_pickup_pos = pickup_pos
                    best_delivery_pos = delivery_pos

        # convert to number rather than time since caller of this function expects that
        if best_delta == dt.timedelta.max:
            best_delta = float('inf')
        else:
            best_delta = best_delta.total_seconds()

        # return the best_delta
        return best_delta, best_pickup_pos, best_delivery_pos


class TimeShiftRegretInsertion(PDPParallelInsertionConstruction):
    # gotta override this method for regret measures
    def best_insertion_for_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                   request: int):
        """For the given request, finds the best combination of (a) tour, (b) pickup position and (c) delivery position
         for the best insertion. Best, in this case, is defined by the inheriting class (e.g. lowest cost increase or
         least time shift).

         :returns: (delta, tour, pickup_position, delivery_position) of the best found insertion
         """
        best_delta = float('inf')
        best_tour = None
        best_pickup_pos = None
        best_delivery_pos = None

        best_insertion_for_request_in_tour = []

        for tour in range(solution.carriers[carrier].num_tours()):

            delta, pickup_pos, delivery_pos = self.best_insertion_for_request_in_tour(instance, solution,
                                                                                      solution.carriers[carrier].tours[
                                                                                          tour], request)
            best_insertion_for_request_in_tour.append((delta, pickup_pos, delivery_pos))

            if delta < best_delta:
                best_delta = delta
                best_tour = tour
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

        if best_delta < float('inf'):
            regret = sum([(delta - best_delta) for delta in [move[0] for move in best_insertion_for_request_in_tour]])
            # return the inverse of the regret, since the LARGEST regret is to be inserted first
            return 1 / regret, best_tour, best_pickup_pos, best_delivery_pos
        else:
            return best_delta, best_tour, best_pickup_pos, best_delivery_pos

    @pr.timing
    def best_insertion_for_request_in_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_,
                                           request: int, check_feasibility=True) -> Tuple[float, int, int]:
        return MinTimeShiftInsertion().best_insertion_for_request_in_tour(instance, solution, tour_, request,
                                                                          check_feasibility)


class TravelDistanceRegretInsertion(PDPParallelInsertionConstruction):
    # gotta override this method for regret measures
    def best_insertion_for_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                   request: int):
        # steal the regret implementation from time shift
        return TimeShiftRegretInsertion().best_insertion_for_request(instance, solution, carrier, request)

    @pr.timing
    def best_insertion_for_request_in_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_,
                                           request: int, check_feasibility=True) -> Tuple[float, int, int]:
        return MinTravelDistanceInsertion().best_insertion_for_request_in_tour(instance, solution, tour_, request,
                                                                               check_feasibility)
