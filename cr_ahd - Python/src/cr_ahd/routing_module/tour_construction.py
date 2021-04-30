import datetime as dt
from abc import ABC, abstractmethod
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
import logging

logger = logging.getLogger(__name__)


class TourConstructionBehavior(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class InsertionConstruction(TourConstructionBehavior):
    def solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            self._carrier_cheapest_insertion(instance, solution, carrier)
        pass

    @abstractmethod
    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        pass

    def _tour_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int,
                                 unrouted_request: int):
        """Find the cheapest insertions for pickup and delivery for a given tour"""
        t: tr.Tour = solution.carrier_solutions[carrier].tours[tour]

        best_delta = float('inf')
        best_pickup_position = None
        best_delivery_position = None

        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(unrouted_request)

        for pickup_pos in range(1, len(t) - 1):
            for delivery_pos in range(pickup_pos + 1, len(t)):
                delta = t.insertion_distance_delta(instance, [pickup_pos, delivery_pos],
                                                   [pickup_vertex, delivery_vertex])
                if not t.insertion_feasibility_check(instance, solution, [pickup_pos, delivery_pos],
                                                     [pickup_vertex, delivery_vertex]):
                    continue
                if delta < best_delta:
                    best_delta = delta
                    best_pickup_position = pickup_pos
                    best_delivery_position = delivery_pos
        return best_delta, best_pickup_position, best_delivery_position

    def _create_new_tour_with_request(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                                      request: int):
        cs = solution.carrier_solutions[carrier]
        if cs.num_tours() >= instance.carriers_max_num_tours:
            logger.error(f'Max Vehicle Constraint violated!')
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Max. number of vehicles is {instance.carriers_max_num_tours}!')
        rtmp = tr.Tour(len(cs.tours), instance, solution, cs.id_)
        rtmp.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        cs.tours.append(rtmp)
        cs.unrouted_requests.remove(request)
        return


class SequentialInsertion(InsertionConstruction):
    """
    For one request at a time, will find its cheapest insertion position based on the routes built so far. Does not
    consider the other request ( != Cheapest Insertion)

    """

    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        raise NotImplementedError
        logger.debug(f'Sequential Insertion tour construction for carrier {carrier}:')
        cs = solution.carrier_solutions[carrier]
        for r in cs.unrouted_requests:
            best_delta = float('inf')
            for t in range(cs.num_tours()):
                # cheapest way to fit r into t
                delta, pickup_position, delivery_position = self._tour_cheapest_insertion()
                if delta < best_delta:
                    best_delta = delta
                    best_pickup_pos = pickup_position
                    best_delivery_pos = delivery_position
                    best_tour = t
            # if no feasible insertion for the current request was found, create a new tour
            if best_delta == float('inf'):
                self._create_new_tour_with_request(instance, solution, carrier, r)
                break
            else:
                cs.tours[best_tour].insert_and_update(instance, solution, [best_pickup_pos, best_delivery_pos],
                                                      instance.pickup_delivery_pair(r))
                cs.unrouted_requests.remove(r)


class CheapestInsertion(InsertionConstruction):
    """
    For each REQUEST, identify its cheapest insertion. Compare the collected insertion costs and insert the cheapest
    over all requests.
    """

    def _carrier_cheapest_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        logger.debug(f'Cheapest Insertion tour construction for carrier {carrier}:')
        cs = solution.carrier_solutions[carrier]
        while cs.unrouted_requests:
            best_delta = float('inf')
            best_request = best_pickup_position = best_delivery_position = best_tour = None
            start_over = False
            for r in cs.unrouted_requests:
                best_delta_for_r = float('inf')
                for t in range(cs.num_tours()):
                    # cheapest way to fit r into t
                    delta, pickup_position, delivery_position = self._tour_cheapest_insertion(instance, solution,
                                                                                              carrier, t, r)
                    if delta < best_delta:
                        best_delta = delta
                        best_request = r
                        best_pickup_position = pickup_position
                        best_delivery_position = delivery_position
                        best_tour = t
                    if delta < best_delta_for_r:
                        best_delta_for_r = delta
                # if no feasible insertion for the current request was found, create a new tour
                if best_delta_for_r == float('inf'):
                    self._create_new_tour_with_request(instance, solution, carrier, r)
                    start_over = True
                    break
            if not start_over:
                cs.tours[best_tour].insert_and_update(instance, solution,
                                                      [best_pickup_position, best_delivery_position],
                                                      instance.pickup_delivery_pair(best_request))
                cs.unrouted_requests.remove(best_request)
        pass


'''
class I1Insertion(TourConstructionBehavior):
    def find_insertion(self, instance: it.PDPInstance, carrier_solution: slt.PDPSolution):
        """
        Find the next optimal Vertex and its optimal insertion position based on the I1 insertion scheme.
        :return: Tuple(u_best, tour_best, rho_best, max_c2)
        """

        tour_best = None
        rho_best = None
        u_best = None
        max_c2 = dt.timedelta.min

        for unrouted in carrier_solution.unrouted_requests:
            # check first the tours that are active (to avoid small tours), if infeasible -> caller must take care!
            for tour in carrier_solution.tours:
                rho, c2 = find_best_feasible_I1_insertion(instance, tour, unrouted)
                if c2 > max_c2:
                    if self.verbose > 1:
                        print(f'^ is the new best c2')
                    tour_best = vehicle
                    rho_best = rho
                    u_best = unrouted
                    max_c2 = c2

        return u_best, tour_best, rho_best, max_c2

    def solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        """
        Use the I1 construction method (Solomon 1987) to build tours for all carriers. If a carrier requires tour
        initialization, it will 'interrupt' and return the corresponding carrier
        """

        logger.debug(f'I1 Construction for {instance}:')
        for carrier_solution in solution.carriers:
            self.solve_carrier(instance, carrier_solution)

        return None

    def solve_carrier(self, instance: it.PDPInstance, carrier_solution: slt.PDPSolution):
        # construction loop
        while carrier_solution.unrouted_requests:
            vertex, vehicle, position, _ = self.find_insertion(instance, carrier_solution)
            if position:  # insert
                vehicle.tour.insert_and_update(index=position, vertex=vertex)
                if self.verbose > 0:
                    print(f'\tInserting {vertex.id_} into {carrier.id_}.{vehicle.tour.id_}')
            else:
                raise ut.InsertionError('', 'no feasible insertion exists for the vehicles that are active already')
        pass
'''

# ===============================================================================
'''
def _c11(tour, i, u, j, mu):
    """weighted insertion cost"""
    c11 = tour.distance_matrix.loc[i.id_, u.id_] + tour.distance_matrix.loc[u.id_, j.id_] - mu * \
          tour.distance_matrix.loc[i.id_, j.id_]
    return ut.travel_time(c11)


def _c12(tour, j_index, u):
    """how much will the start of service of vertex at index j be pushed back? (Given in distance not time!)"""
    service_start_j = tour.service_schedule[j_index]
    tour.insert_and_update(index=j_index, vertex=u)
    service_start_j_new = tour.service_schedule[j_index + 1]
    c12 = service_start_j_new - service_start_j
    tour.pop_and_update(j_index)
    return c12


def _c1(tour, i_index: int, u, j_index: int, alpha_1: float, mu: float, ) -> float:
    """
    c1 criterion of Solomon's I1 insertion heuristic: "best feasible insertion cost"
    Does NOT include a feasibility check. Following the
    description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
    Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
    10.1287/trsc.1030.0056.)
    """
    alpha_2 = 1 - alpha_1
    i = tour.routing_sequence[i_index]
    j = tour.routing_sequence[j_index]
    c11 = _c11(tour, i, u, j, mu)  # cost of insertion
    c12 = _c12(tour, j_index, u)  # cost of arrival "time" postponement at j
    return alpha_1 * c11 + alpha_2 * c12


def _c2(tour, u, c1: float, lambda_: float = ut.opts['lambda'], ):
    """
    c2 criterion of Solomon's I1 insertion heuristic: "find the best customer/request"
    Does NOT include a feasibility check. Following the
    description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
    Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
    10.1287/trsc.1030.0056.)
    """
    return lambda_ * ut.travel_time(tour.distance_matrix.loc[tour.depot.id_, u.id_]) - c1


def find_best_feasible_I1_insertion(instance: it.PDPInstance, tour: tr.Tour, u: int):
    """
    returns float('-inf') if no feasible insertion position was found
    :param instance:
    :param tour:
    :param u:
    :return:
    """
    position_best = None
    max_c2 = dt.timedelta.min
    for position in range(1, len(tour)):
        i = tour.routing_sequence[position - 1]
        j = tour.routing_sequence[position]

        # trivial feasibility check
        if i.tw.e < u.tw.l and u.tw.e < j.tw.l:
            # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
            # compute c1(=best feasible insertion cost) and c2(=best request) and update their best values
            try:
                c1 = _c1(tour,
                         i_index=position - 1,
                         u=u,
                         j_index=position,
                         alpha_1=ut.opts['alpha_1'],
                         mu=ut.opts['mu'],
                         )
            except ut.InsertionError:
                continue
            c2 = _c2(tour=tour, u=u, lambda_=ut.opts['lambda'], c1=c1)
            if c2 > max_c2:
                max_c2 = c2
                position_best = position
    return position_best, max_c2
'''
