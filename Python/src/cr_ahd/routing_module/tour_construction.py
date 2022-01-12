import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

from core_module import instance as it, solution as slt, tour as tr
from utility_module.errors import ConstraintViolationError

logger = logging.getLogger(__name__)
'''
# Probably not worth the effort to build a superclass. I will only use delivery anyways and using loops to iterate
#  over single element (delivery_vertex) lists is just unnecessary overhead

class InsertionConstruction(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def insert_single_request(self, 
                              instance: it.MDVRPTWInstance, 
                              solution: slt.CAHDSolution,
                              carrier_id: int, request: int):
        """
        Inserts a single unrouted request and inserts its vertices in the best position according
        to the criterion of the subclass
        
        :param instance: 
        :param solution: 
        :param carrier_id: 
        :param request: 
        :return: 
        """
        carrier = solution.carriers[carrier_id]
        insertion_criteria, tour, insertion_positions = self.best_insertion_for_request(instance, carrier, request)

        if tour is None:
            self.create_new_tour_with_request(instance, solution, carrier_id, request)
        else:
            self.execute_insertion(instance, solution, carrier_id, request, tour.id_, insertion_positions)
     '''


class VRPTWInsertionConstruction(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def insert_single_request(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                              carrier_id: int, request: int):
        carrier = solution.carriers[carrier_id]
        insertion_criteria, tour, delivery_pos = self.best_insertion_for_request(instance, carrier, request)

        if tour is None:
            self.create_new_tour_with_request(instance, solution, carrier_id, request)
        else:
            self.execute_insertion(instance, solution, carrier_id, request, tour.id_, delivery_pos)

    def best_insertion_for_carrier(self,
                                   instance: it.MDVRPTWInstance,
                                   carrier: slt.AHDSolution) -> Tuple[Optional[int],
                                                                      Optional[tr.Tour],
                                                                      Optional[int]]:
        """
        Scanning through all the unrouted requests of the given carrier, the best one is identified and returned as
        a tuple of (request, tour, delivery_pos). "Best" in this case is defined by the inheriting class
        (e.g. lowest distance increase or smallest time shift)

        :param instance:
        :param carrier:
        :return: the best found insertion as a tuple of (request, tour, delivery_pos)
        """
        logger.debug(f'Cheapest insertion tour construction for carrier {carrier.id_}')

        best_delta = float('inf')
        best_request: Union[None, int] = None
        best_tour: Union[None, tr.Tour] = None
        best_delivery_pos: Union[None, int] = None

        for request in carrier.unrouted_requests:

            delta, tour, delivery_pos = self.best_insertion_for_request(instance, carrier, request)

            if delta < best_delta:
                best_delta = delta
                best_request = request
                best_tour = tour
                best_delivery_pos = delivery_pos

            # if no feasible insertion was found return None for tour
            if best_delta == float('inf'):
                return request, None, None

        return best_request, best_tour, best_delivery_pos

    def best_insertion_for_request(self,
                                   instance: it.MDVRPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int]:
        """

        :param instance:
        :param carrier:
        :param request:
        :return: best insertion as a tuple of (delta, tour, delivery_pos)
        """
        best_delta: float = float('inf')
        best_tour: tr.Tour = None
        best_delivery_pos: int = None

        for tour in carrier.tours:

            delta, delivery_pos = self.best_insertion_for_request_in_tour(instance, tour, request)
            if delta < best_delta:
                best_delta = delta
                best_tour = tour
                best_delivery_pos = delivery_pos

        return best_delta, best_tour, best_delivery_pos

    @abstractmethod
    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour,
                                           request: int, check_feasibility=True) -> Tuple[float, int]:
        pass

    @staticmethod
    def execute_insertion(instance: it.MDVRPTWInstance,
                          solution: slt.CAHDSolution,
                          carrier_id: int,
                          request: int,
                          tour_id: int,
                          delivery_pos: int):
        carrier = solution.carriers[carrier_id]
        tour = solution.tours[tour_id]
        delivery = instance.vertex_from_request(request)
        tour.insert_and_update(instance, [delivery_pos], [delivery])
        tour.requests.add(request)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)

    @staticmethod
    def execute_insertion_in_tour(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, tour_id: int,
                                  request: int, delivery_pos: int):

        delivery = instance.vertex_from_request(request)
        tour = solution.tours[tour_id]
        tour.insert_and_update(instance, [delivery_pos], [delivery])
        tour.requests.add(request)

    def create_pendulum_tour_for_infeasible_request(self,
                                                    instance: it.MDVRPTWInstance,
                                                    solution: slt.CAHDSolution,
                                                    carrier_id: int,
                                                    request: int,
                                                    ):
        carrier = solution.carriers[carrier_id]
        pendulum_tour_id = solution.get_free_pendulum_tour_id()
        pendulum_tour = tr.VRPTWTour(pendulum_tour_id, depot_index=carrier.id_)

        if pendulum_tour.insertion_feasibility_check(instance, [1], [instance.vertex_from_request(request)]):
            pendulum_tour.insert_and_update(instance, [1], [instance.vertex_from_request(request)])
            pendulum_tour.requests.add(request)

        else:
            raise ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier.id_}')

        if int(pendulum_tour_id[1]) < len(solution.tours_pendulum):
            solution.tours_pendulum[int(pendulum_tour_id[1])] = pendulum_tour
        else:
            solution.tours_pendulum.append(pendulum_tour)
        carrier.tours_pendulum.append(pendulum_tour)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)
        pass

    def create_new_tour_with_request(self,
                                     instance: it.MDVRPTWInstance,
                                     solution: slt.CAHDSolution,
                                     carrier_id: int,
                                     request: int):
        carrier = solution.carriers[carrier_id]
        if len(carrier.tours) >= instance.carriers_max_num_tours:
            raise ConstraintViolationError(
                f'Cannot create new route with request {request} for carrier {carrier.id_}.'
                f' Max. number of vehicles is {instance.carriers_max_num_tours}!'
                f' ({instance.id_})')
        tour_id = solution.get_free_tour_id()
        assert tour_id < instance.num_carriers * instance.carriers_max_num_tours, f'{instance.id_}: tour_id={tour_id}'
        logger.debug(f'Carrier {carrier_id}, *Tour {tour_id}')
        tour = tr.VRPTWTour(tour_id, depot_index=carrier.id_)

        if tour.insertion_feasibility_check(instance, [1], [instance.vertex_from_request(request)]):
            tour.insert_and_update(instance, [1, 2], [instance.vertex_from_request(request)])
            tour.requests.add(request)

        else:
            raise ConstraintViolationError(
                f'{instance.id_} Cannot create new route with request {request} for carrier {carrier.id_}.')

        if tour_id < len(solution.tours):
            solution.tours[tour_id] = tour
        else:
            solution.tours.append(tour)
        carrier.tours.append(tour)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)
        return


class VRPTWMinTravelDistanceInsertion(VRPTWInsertionConstruction):

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.VRPTWTour,
                                           request: int, check_feasibility=True) -> Tuple[float, int]:
        logger.warning('Distance of vienna instances violates triangle inequality!')
        delivery_vertex = instance.vertex_from_request(request)
        best_delta = float('inf')
        best_delivery_position = None

        for delivery_pos in range(1, len(tour)):
            delta = tour.insert_distance_delta(instance, [delivery_pos], [delivery_vertex])
            if delta < best_delta:

                update_best = True
                if check_feasibility:
                    update_best = tour.insertion_feasibility_check(instance, [delivery_pos], [delivery_vertex])

                if update_best:
                    best_delta = delta
                    best_delivery_position = delivery_pos

        return best_delta, best_delivery_position


class VRPTWMinTravelDurationInsertion(VRPTWInsertionConstruction):

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.VRPTWTour,
                                           request: int, check_feasibility=True) -> Tuple[float, int]:
        delivery_vertex = instance.vertex_from_request(request)
        best_delta = float('inf')
        best_delivery_position = None

        for delivery_pos in range(1, len(tour)):
            delta = tour.insert_duration_delta(instance, [delivery_pos], [delivery_vertex]).total_seconds()
            if delta < best_delta:

                update_best = True
                if check_feasibility:
                    update_best = tour.insertion_feasibility_check(instance, [delivery_pos], [delivery_vertex])

                if update_best:
                    best_delta = delta
                    best_delivery_position = delivery_pos

        return best_delta, best_delivery_position


'''
class PDPParallelInsertionConstruction(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def insert_all_unrouted_statically(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier_id: int):
        """
        Inserts all unrouted requests of the specified carrier into tours. Repeatedly finds the "best" insertion
        which is defined by the request, the tour and the insertion positions. "best" is defined by the inheriting
        classes.
        """
        carrier = solution.carriers[carrier_id]
        while carrier.unrouted_requests:
            request, tour, pickup_pos, delivery_pos = self.best_insertion_for_carrier(instance, carrier)

            # when for a given request no tour can be found, create a new tour and start over
            # this will fail if max_num_vehicles is exceeded
            if tour is None:
                self.create_new_tour_with_request(instance, solution, carrier_id, request)

            # otherwise insert as suggested
            else:
                self.execute_insertion(instance, solution, carrier_id, request, tour.id_, pickup_pos, delivery_pos)

        pass

    def insert_all_unrouted_dynamically(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                                        carrier_id: int):
        """
        Inserts all unrouted requests of the specified carrier into tours. Repeatedly takes the first request in the
        list of unrouted requests and inserts it into its "best" position. "best" is defined by the inheriting classes.
        """
        raise NotImplementedError(f'This has never been used before, check before removing the Error raise')
        carrier = solution.carriers[carrier_id]
        while carrier.unrouted_requests:
            request = carrier.unrouted_requests[0]
            self.insert_single_request(instance, solution, carrier_id, request)
        pass

    def insert_single_request(self,
                              instance: it.MDVRPTWInstance,
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
            self.create_new_tour_with_request(instance, solution, carrier_id, request)
        else:
            self.execute_insertion(instance, solution, carrier_id, request, tour.id_, pickup_pos, delivery_pos)

    def best_insertion_for_carrier(self,
                                   instance: it.MDVRPTWInstance,
                                   carrier: slt.AHDSolution) -> Tuple[Optional[int],
                                                                      Optional[tr.Tour],
                                                                      Optional[int],
                                                                      Optional[int]]:
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
                                   instance: it.MDVRPTWInstance,
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
    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        """
        returns the (feasible) insertion move with the lowest cost as a tuple of (delta, pickup_position,
        delivery_position)

        :param check_feasibility:
        """
        pass

    @staticmethod
    def execute_insertion(instance: it.MDVRPTWInstance,
                          solution: slt.CAHDSolution,
                          carrier_id: int,
                          request: int,
                          tour_id: int,
                          pickup_pos: int,
                          delivery_pos: int):
        carrier = solution.carriers[carrier_id]
        tour = solution.tours[tour_id]
        pickup, delivery = instance.pickup_delivery_pair(request)
        logger.debug(f'Carrier {carrier_id}, Tour {tour_id}, request {request}{pickup, delivery} insertion')
        tour.insert_and_update(instance, [pickup_pos, delivery_pos], [pickup, delivery])
        tour.requests.add(request)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)

    @staticmethod
    def execute_insertion_in_tour(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, tour_id: int,
                                  request: int, pickup_pos: int, delivery_pos: int):

        pickup, delivery = instance.pickup_delivery_pair(request)
        tour = solution.tours[tour_id]
        tour.insert_and_update(instance, [pickup_pos, delivery_pos], [pickup, delivery])
        tour.requests.add(request)

    def create_pendulum_tour_for_infeasible_request(self,
                                                    instance: it.MDVRPTWInstance,
                                                    solution: slt.CAHDSolution,
                                                    carrier_id: int,
                                                    request: int
                                                    ):
        carrier = solution.carriers[carrier_id]
        pendulum_tour_id = solution.get_free_pendulum_tour_id()
        logger.debug(f'Carrier {carrier_id}, *Pendulum Tour {pendulum_tour_id}')
        pendulum_tour = tr.Tour(pendulum_tour_id, depot_index=carrier.id_)

        if pendulum_tour.insertion_feasibility_check(instance, [1, 2], instance.pickup_delivery_pair(request)):
            pendulum_tour.insert_and_update(instance, [1, 2], instance.pickup_delivery_pair(request))
            pendulum_tour.requests.add(request)

        else:
            raise ConstraintViolationError(
                f'Cannot create new route with request {request} for carrier {carrier.id_}.')

        if int(pendulum_tour_id[1]) < len(solution.tours_pendulum):
            solution.tours_pendulum[int(pendulum_tour_id[1])] = pendulum_tour
        else:
            solution.tours_pendulum.append(pendulum_tour)
        carrier.tours_pendulum.append(pendulum_tour)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)
        return

    def create_new_tour_with_request(self,
                                     instance: it.MDVRPTWInstance,
                                     solution: slt.CAHDSolution,
                                     carrier_id: int,
                                     request: int):
        carrier = solution.carriers[carrier_id]
        if len(carrier.tours) >= instance.carriers_max_num_tours:
            raise ConstraintViolationError(
                f'Cannot create new route with request {request} for carrier {carrier.id_}.'
                f' Max. number of vehicles is {instance.carriers_max_num_tours}!'
                f' ({instance.id_})')
        tour_id = solution.get_free_tour_id()
        assert tour_id < instance.num_carriers * instance.carriers_max_num_tours, f'{instance.id_}: tour_id={tour_id}'
        logger.debug(f'Carrier {carrier_id}, *Tour {tour_id}')
        tour = tr.Tour(tour_id, depot_index=carrier.id_)

        if tour.insertion_feasibility_check(instance, [1, 2], instance.pickup_delivery_pair(request)):
            tour.insert_and_update(instance, [1, 2], instance.pickup_delivery_pair(request))
            tour.requests.add(request)

        else:
            raise ConstraintViolationError(
                f'{instance.id_} Cannot create new route with request {request} for carrier {carrier.id_}.')

        if tour_id < len(solution.tours):
            solution.tours[tour_id] = tour
        else:
            solution.tours.append(tour)
        carrier.tours.append(tour)
        carrier.unrouted_requests.remove(request)
        carrier.routed_requests.append(request)
        return


class MinTravelDistanceInsertion(PDPParallelInsertionConstruction):
    """
    For each request, identify its cheapest insertion based on distance delta. Compare the collected insertion costs
    and insert the cheapest over all requests.
    """

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour, request: int,
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

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour, request: int,
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
                                   instance: it.MDVRPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int, int]:
        """
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

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        return MinTimeShiftInsertion().best_insertion_for_request_in_tour(instance, tour, request, check_feasibility)


class TravelDistanceRegretInsertion(PDPParallelInsertionConstruction):
    # gotta override this method for regret measures
    def best_insertion_for_request(self,
                                   instance: it.MDVRPTWInstance,
                                   carrier: slt.AHDSolution,
                                   request: int) -> Tuple[float, tr.Tour, int, int]:
        # steal the regret implementation from time shift
        return TimeShiftRegretInsertion().best_insertion_for_request(instance, carrier, request)

    def best_insertion_for_request_in_tour(self, instance: it.MDVRPTWInstance, tour: tr.Tour, request: int,
                                           check_feasibility=True) -> Tuple[float, int, int]:
        return MinTravelDistanceInsertion().best_insertion_for_request_in_tour(instance, tour, request,
                                                                               check_feasibility)
'''
