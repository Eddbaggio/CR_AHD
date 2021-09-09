import datetime as dt
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class PDPParallelInsertionConstruction(ABC):
    def insert_all(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier_id: int):
        """inserts ALL unrouted requests of the specified carrier"""
        carrier = solution.carriers[carrier_id]
        while carrier.unrouted_requests:
            request, tour, pickup_pos, delivery_pos = self.best_insertion_for_carrier(instance, carrier)

            # when for a given request no tour can be found, create a new tour and start over
            # this will fail if max_num_vehicles is exceeded
            if tour is None:
                self.create_new_tour_with_request(instance, solution.num_tours(), solution, carrier_id, request)

            # otherwise insert as suggested
            else:
                self.execute_insertion(instance, solution, carrier_id, request, tour.id_, pickup_pos, delivery_pos)

        pass

    def insert_single(self,
                      instance: it.MDPDPTWInstance,
                      solution: slt.CAHDSolution,
                      carrier_id: int,
                      request: int):
        """
        Take a single (!) unrouted customer of the given carrier and insert it in the best position.\n
        NOTE: Function alters the solution in place
        """
        carrier = solution.carriers[carrier_id]
        insertion_criteria, tour, pickup_pos, delivery_pos = self.best_insertion_for_request(instance, carrier, request)

        if tour is None:
            self.create_new_tour_with_request(instance, solution.num_tours(), solution, carrier_id, request)
        else:
            self.execute_insertion(instance, solution, carrier_id, request, tour.id_, pickup_pos, delivery_pos)

        # REMOVEME print insertions of *original* requests
        # if all([x in range(carrier_id*instance.num_requests_per_carrier, carrier_id*instance.num_requests_per_carrier+instance.num_requests_per_carrier) for x in carrier.accepted_requests]):
        #     if tour is None:
        #         print(f'Request {request:02d} insertion: *tour {solution.num_tours()-1:02d}, pickup_pos: 01, delivery_pos: 02, sum_profit: {carrier.sum_profit()}')
        #     else:
        #         print(f'Request {request:02d} insertion:  tour {tour.id_:02d}, pickup_pos: {pickup_pos:02d}, delivery_pos: {delivery_pos:02d}, sum_profit: {carrier.sum_profit()}')
        # pass

    def best_insertion_for_carrier(self,
                                   instance: it.MDPDPTWInstance,
                                   carrier: slt.AHDSolution) -> Tuple[Union[None, int],
                                                                      Union[None, tr.Tour],
                                                                      Union[None, int],
                                                                      Union[None, int]]:
        """
        Scanning through all the unrouted requests of the given carrier, the best one is identified and returned as
        a tuple of (request, tour, pickup_pos, delivery_pos). "Best" in this case is defined by the inheriting class
        (e.g. lowest distance increase or smallest time shift)

        :return: the best found insertion as a tuple of (request, tour, pickup_pos, delivery_pos)
        """
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier.id_}:')

        best_delta = float('inf')
        best_request: Union[None, int] = None
        best_tour: Union[None, tr.Tour] = None
        best_pickup_pos: Union[None, int] = None
        best_delivery_pos: Union[None, int] = None

        for request in carrier.unrouted_requests:

            delta, tour, pickup_pos, delivery_pos = self.best_insertion_for_request(instance, carrier, request)

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

    def best_insertion_for_request(self,
                                   instance: it.MDPDPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int, int]:
        """For the given request, finds the best combination of (a) tour, (b) pickup position and (c) delivery position
         for the best insertion. Best, in this case, is defined by the inheriting class (e.g. lowest cost increase or
         least time shift).

         :returns: delta, tour, pickup_position, delivery_position of the best found insertion
         """
        best_delta: float = float('inf')
        best_tour: tr.Tour = None
        best_pickup_pos: int = None
        best_delivery_pos: int = None

        for tour in carrier.tours:

            delta, pickup_pos, delivery_pos = self.best_insertion_for_request_in_tour(instance, tour, request)
            if delta < best_delta:
                best_delta = delta
                best_tour = tour
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

        return best_delta, best_tour, best_pickup_pos, best_delivery_pos

    @abstractmethod
    def best_insertion_for_request_in_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        """
        returns the (feasible) insertion move with the lowest cost as a tuple of (delta, pickup_position,
        delivery_position)

        :param check_feasibility:
        """
        pass

    @staticmethod
    def execute_insertion(instance: it.MDPDPTWInstance,
                          solution: slt.CAHDSolution,
                          carrier_id: int,
                          request: int,
                          tour_id: int,
                          pickup_pos: int,
                          delivery_pos: int):
        carrier = solution.carriers[carrier_id]
        tour = solution.tours[tour_id]
        pickup, delivery = instance.pickup_delivery_pair(request)
        tour.insert_and_update(instance, [pickup_pos, delivery_pos], [pickup, delivery])
        tour.requests.add(request)
        solution.request_to_tour_assignment[request] = tour.id_  # TODO extract this for solution-independence?
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)

    @staticmethod
    def execute_insertion_in_tour(instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, tour_id: int,
                                  request: int, pickup_pos: int, delivery_pos: int):

        pickup, delivery = instance.pickup_delivery_pair(request)
        tour = solution.tours[tour_id]
        tour.insert_and_update(instance, [pickup_pos, delivery_pos], [pickup, delivery])
        solution.request_to_tour_assignment[instance.request_from_vertex(pickup)] = tour.id_

    def create_new_tour_with_request(self,
                                     instance: it.MDPDPTWInstance,
                                     tour_id: int,
                                     solution: slt.CAHDSolution,
                                     carrier_id: int,
                                     request: int):
        carrier = solution.carriers[carrier_id]
        if carrier.num_tours() >= instance.carriers_max_num_tours:
            raise ut.ConstraintViolationError(
                f'Cannot create new route with request {request} for carrier {carrier.id_}.'
                f' Max. number of vehicles is {instance.carriers_max_num_tours}!'
                f' ({instance.id_})')
        tour = tr.Tour(tour_id, depot_index=carrier.id_)

        if tour.insertion_feasibility_check(instance, [1, 2], instance.pickup_delivery_pair(request)):
            tour.insert_and_update(instance, [1, 2], instance.pickup_delivery_pair(request))
            tour.requests.add(request)
            solution.request_to_tour_assignment[request] = tour.id_  # TODO extract this for solution-independence?

        else:
            raise ut.ConstraintViolationError(
                f'Cannot create new route with request {request} for carrier {carrier.id_}.')

        solution.tours.append(tour)  # TODO extract this for solution-independence?
        carrier.tour_ids.append(tour.id_)
        carrier.tours.append(tour)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)
        return


class MinTravelDistanceInsertion(PDPParallelInsertionConstruction):
    """
    For each request, identify its cheapest insertion based on distance delta. Compare the collected insertion costs
    and insert the cheapest over all requests.
    """

    def best_insertion_for_request_in_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        best_delta = float('inf')
        best_pickup_position = None
        best_delivery_position = None

        for pickup_pos in range(1, len(tour)):
            for delivery_pos in range(pickup_pos + 1, len(tour) + 1):
                delta = tour.insert_distance_delta(instance, [pickup_pos, delivery_pos],
                                                   [pickup_vertex, delivery_vertex])
                if delta < best_delta:

                    update_best = True
                    if check_feasibility:
                        update_best = tour.insertion_feasibility_check(instance, [pickup_pos, delivery_pos],
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

    def best_insertion_for_request_in_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        """Find the insertions for pickup and delivery for a given tour that have the best C value
        :param check_feasibility:
        :param tour:
        """

        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        best_delta = dt.timedelta.max
        best_pickup_pos = None
        best_delivery_pos = None

        for pickup_pos in range(1, len(tour)):

            for delivery_pos in range(pickup_pos + 1, len(tour) + 1):
                # c1 + c2 + c3 = max_shift_delta; i.e. the decrease in max_shift due to the insertions
                delta = tour.insert_max_shift_delta(instance, [pickup_pos, delivery_pos],
                                                    [pickup_vertex, delivery_vertex])

                update_best = True
                if check_feasibility:
                    update_best = tour.insertion_feasibility_check(instance, [pickup_pos, delivery_pos],
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
    def best_insertion_for_request(self,
                                   instance: it.MDPDPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int, int]:
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

        for tour in carrier.tours:

            delta, pickup_pos, delivery_pos = self.best_insertion_for_request_in_tour(instance, tour, request)
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

    def best_insertion_for_request_in_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        return MinTimeShiftInsertion().best_insertion_for_request_in_tour(instance, tour, request, check_feasibility)


class TravelDistanceRegretInsertion(PDPParallelInsertionConstruction):
    # gotta override this method for regret measures
    def best_insertion_for_request(self,
                                   instance: it.MDPDPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int, int]:
        # steal the regret implementation from time shift
        return TimeShiftRegretInsertion().best_insertion_for_request(instance, carrier, request)

    def best_insertion_for_request_in_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        return MinTravelDistanceInsertion().best_insertion_for_request_in_tour(instance, tour, request,
                                                                               check_feasibility)
