from abc import ABC, abstractmethod

from src.cr_ahd.utility_module.profiling import Timer
from src.cr_ahd.utility_module.utils import opts, InsertionError, get_carrier_by_id


class TourConstructionBehavior(ABC):
    """
    Visitor Interface to apply a tour construction heuristic to either an instance (i.e. each of its carriers)
    or a single specific carrier. Contrary to the initializer this visitor will allocate all requests
    """

    def __init__(self, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """
        :param verbose:
        :param plot_level:
        """

        self.verbose = verbose
        self.plot_level = plot_level
        self._runtime = 0
        # TODO: add timer for profiling?

    # could these solve methods be templates? there are many steps that are identical, e.g. setting instance.solved=True
    @abstractmethod
    def solve_instance(self, instance):
        pass

    @abstractmethod
    def solve_carrier(self, carrier):
        pass

    # @abstractmethod
    # def find_insertion(self, carrier, vertex=None):
    #     pass


class SequentialCheapestInsertion(TourConstructionBehavior):
    """
    For one customer at a time, will iterate over requests in their order of assignment and finds their cheapest
    insertion based on the route built so far.

    """

    def find_insertion(self, carrier, vertex):
        """
        Checks all active vehicles/tours and one inactive vehicle/tour for a feasible insertion and return the cheapest
        one. This Insertion method is used by some other RoutingVisitors, too.

        :param vertex: Vertex to be inserted
        :param carrier: carrier which shall insert the vertex into his routes
        :return: triple (vehicle_best, position_best, cost_best) defining the cheapest vehicle/tour, index and
        associated cost to insert the given vertex u
        """
        vehicle_best = None
        cost_best = float('inf')
        position_best = None
        # check all active vehicles and the first INACTIVE one for insertion cost
        for vehicle in [*carrier.active_vehicles, carrier.inactive_vehicles[0]]:
            try:
                # TODO: ugly solution to have a standalone function for this:
                position, cost = find_cheapest_feasible_insertion(vehicle.tour, request=vertex)
                if self.verbose > 1:
                    print(f'\t\tInsertion cost {vertex.id_} -> {vehicle.id_}: {cost}')
                if cost < cost_best:  # a new cheapest insertion was found -> update the incumbents
                    vehicle_best = vehicle
                    cost_best = cost
                    position_best = position
            except InsertionError:
                if self.verbose > 0:
                    print(f'x\t{vertex.id_} cannot be feasibly inserted into {vehicle.id_}')
        return vehicle_best, position_best, cost_best

    def solve_instance(self, instance):
        """
        Use a static construction method to build tours for all carriers via SEQUENTIAL CHEAPEST INSERTION.
        (Can also be used for dynamic route construction if the request-to-carrier assignment is known.)
        """
        # assert not instance.solved, f'Instance {instance} has already been solved'

        if self.verbose > 0:
            print(f'Cheapest Insertion Construction for {instance}:')
        timer = Timer()
        timer.start()

        while any(instance.assigned_unrouted_requests()):
            vertex = instance.assigned_unrouted_requests()[0]
            carrier = get_carrier_by_id(instance.carriers, vertex.carrier_assignment)
            vehicle, position, _ = self.find_insertion(carrier, vertex)
            vehicle.tour.insert_and_update(index=position, vertex=vertex)

        timer.stop()
        self._runtime = timer.duration
        # instance.routing_visitor = self
        # instance.solved = True
        pass

    def solve_carrier(self, carrier):
        pass


class I1Insertion(TourConstructionBehavior):
    def find_insertion(self, carrier):
        """
        Find the next optimal Vertex and its optimal insertion position based on the I1 insertion scheme.
        :return: Tuple(u_best, vehicle_best, rho_best, max_c2)
        """

        vehicle_best = None
        rho_best = None
        u_best = None
        max_c2 = float('-inf')

        for unrouted in carrier.unrouted_requests:  # take the unrouted requests
            # check first the vehicles that are active (to avoid small tours), if infeasible -> caller must take care!
            for vehicle in carrier.active_vehicles:
                # TODO tour.find_best_feasible_I1_insertion() must be moved from tour class to (the tour's?) visitor!
                rho, c2 = find_best_feasible_I1_insertion(vehicle.tour, unrouted)
                if c2 > max_c2:
                    if self.verbose > 1:
                        print(f'^ is the new best c2')
                    vehicle_best = vehicle
                    rho_best = rho
                    u_best = unrouted
                    max_c2 = c2

        return u_best, vehicle_best, rho_best, max_c2

    def solve_instance(self, instance):
        """
        Use the I1 construction method (Solomon 1987) to build tours for all carriers. If a carrier requires tour
        initialization, it will 'interrupt' and return the corresponding carrier
        """
        # assert not instance.solved, f'Instance {instance} has already been solved'
        # ensure that tours have been initialized as is done in the original I1 algorithm
        # assert instance.initialized, f'Tours must be initialized before solving with I1'

        if self.verbose > 0:
            print(f'STATIC I1 Construction for {instance}:')
        timer = Timer()
        timer.start()

        for carrier in instance.carriers:
            # if self.plot_level > 1:
            #     ani = CarrierConstructionAnimation(
            #         carrier,
            #         f"{instance.id_}{' centralized' if instance.num_carriers == 0 else ''}:" \
            #         f" Solomon's I1 construction: {carrier.id_}")
            try:
                self.solve_carrier(carrier)
            except InsertionError:
                timer.stop()
                self._runtime = timer.duration
                return carrier

        timer.stop()
        self._runtime = timer.duration
        # instance.routing_visitor = self
        # instance.solved = True
        return None

    def solve_carrier(self, carrier):
        # construction loop
        while carrier.unrouted_requests:
            vertex, vehicle, position, _ = self.find_insertion(carrier)
            if position:  # insert
                vehicle.tour.insert_and_update(index=position, vertex=vertex)
                if self.verbose > 0:
                    print(f'\tInserting {vertex.id_} into {carrier.id_}.{vehicle.tour.id_}')
            else:
                raise InsertionError('', 'no feasible insertion exists for the vehicles that are active already')

        if self.verbose > 0:
            print(f'Total Route cost of carrier {carrier.id_}: {carrier.cost()}\n')
        # carrier.routing_visitor = self
        # carrier.solved = True
        pass


'''
class DynamicInsertionWithAuction(RoutingVisitor):
    """Like DynamicInsertion but also has an auction after each request"""

    def find_insertion(self, carrier, vertex):
        """currently uses static cheapest insertion to find the position"""
        vehicle_best, position_best, cost_best = \
            SequentialCheapestInsertion(self.verbose, self.plot_level).find_insertion(carrier, vertex)
        return vehicle_best, position_best, cost_best

    def solve_instance(self, instance):
        # assert not instance.solved, f'Instance {instance} has already been solved with {instance.routing_visitor.__class__.__name__}'

        if self.verbose > 0:
            print(f'DYNAMIC Cheapest Insertion Construction WITH AUCTION for {instance}:')

        timer = Timer()
        timer.start()

        # TODO this function assumes that requests have been assigned to
        #  carriers already which is not really logical
        #  in a real-life case since they arrive dynamically
        # for carrier in instance.carriers:
        #     carrier.routing_visitor = SequentialCheapestInsertion()
        for request in instance.requests:  # TODO: this will also include the requests that have been chosen for initiialization. Initialization does not make any sense for Dynamic Problems, so just iterating over the unrouted does not solve the issue
            preliminary_carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)

            # do the auction, i.e. find the (global) insertion
            # TODO auction stuff shouldn't be a member function. Try strategy or visitor pattern or template
            #  method instead
            carrier, vehicle, position, cost = instance.cheapest_insertion_auction(
                request=request,
                initial_carrier=preliminary_carrier)
            if preliminary_carrier != carrier:
                preliminary_carrier.retract_request(request)
                carrier.assign_request(request)  # assign to auction winner

            # attempt insertion
            try:
                if self.verbose > 0:
                    print(f'\tInserting {request.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_update(position, request)
            except TypeError:
                raise InsertionError('', f"Cannot insert {request} feasibly into {carrier.id_}.{vehicle.id_}")

        timer.stop()
        self._runtime = timer.duration
        # instance.solved = True
        # instance.routing_visitor = self
        pass

    def solve_carrier(self, carrier):
        raise NotImplementedError
        pass
'''


# ===============================================================================


def find_cheapest_feasible_insertion(tour, request, verbose=opts['verbose']):
    """
    :return: Tuple (position, cost) of the best insertion position index and the associated (lowest) cost
    """
    assert request.routed is False, f'Trying to insert an already routed vertex! {request}'
    if verbose > 2:
        print(f'\n= Cheapest insertion of {request.id_} into {tour.id_}')
        print(tour)
    min_insertion_cost = float('inf')
    i_best = None
    j_best = None

    # test all insertion positions
    for rho in range(1, len(tour)):
        i = tour.routing_sequence[rho - 1]
        j = tour.routing_sequence[rho]

        # trivial feasibility checks
        # if request.tw.e > j.tw.l:
        #     break
        if i.tw.e < request.tw.l and request.tw.e < j.tw.l:  # todo does not consider service times
            insertion_cost = tour.insertion_cost_no_fc(i, j, request)
            if verbose > 2:
                print(f'Between {i.id_} and {j.id_}: {insertion_cost}')
            if insertion_cost < min_insertion_cost:
                try:
                    tour.insert_and_update(index=rho, vertex=request)  # may throw an InsertionError
                    min_insertion_cost = insertion_cost
                    i_best = i
                    j_best = j
                    insertion_position = rho
                    tour.pop_and_update(index=rho)
                except InsertionError:
                    continue
    # return the best found position and cost or raise an error if no feasible position was found
    if i_best:
        if verbose > 2:
            print(f'== Best: between {i_best.id_} and {j_best.id_}: {min_insertion_cost}')
        return insertion_position, min_insertion_cost
    else:
        raise InsertionError('', 'No feasible insertion position found')


# ===================================


def _c11(tour, i, u, j, mu):
    """weighted insertion cost"""
    c11 = tour.distance_matrix.loc[i.id_, u.id_] + tour.distance_matrix.loc[u.id_, j.id_] - mu * \
          tour.distance_matrix.loc[i.id_, j.id_]
    return c11


def _c12(tour, j_index, u):
    """how much will the arrival time of vertex at index j be pushed back?"""
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
    c11 = _c11(tour, i, u, j, mu)
    c12 = _c12(tour, j_index, u)
    return alpha_1 * c11 + alpha_2 * c12


def _c2(tour, u, c1: float, lambda_: float = opts['lambda'], ):
    """
    c2 criterion of Solomon's I1 insertion heuristic: "find the best customer/request"
    Does NOT include a feasibility check Following the
    description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
    Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
    10.1287/trsc.1030.0056.)
    """
    return lambda_ * tour.distance_matrix.loc[tour.depot.id_, u.id_] - c1


def find_best_feasible_I1_insertion(tour, u, verbose=opts['verbose']):
    """
    returns float('-inf') if no feasible insertion position was found
    :param tour:
    :param u:
    :param verbose:
    :return:
    """
    rho_best = None
    max_c2 = float('-inf')
    for rho in range(1, len(tour)):
        i = tour.routing_sequence[rho - 1]
        j = tour.routing_sequence[rho]

        # trivial feasibility check
        if i.tw.e < u.tw.l and u.tw.e < j.tw.l:
            # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
            # compute c1(=best feasible insertion cost) and c2(=best request) and update their best values
            try:
                c1 = _c1(tour,
                         i_index=rho - 1,
                         u=u,
                         j_index=rho,
                         alpha_1=opts['alpha_1'],
                         mu=opts['mu'],
                         )
            except InsertionError:
                continue
            if verbose > 1:
                print(f'c1({u.id_}->{tour.id_}): {c1}')
            c2 = _c2(tour=tour, u=u, lambda_=opts['lambda'], c1=c1)
            if verbose > 1:
                print(f'c2({u.id_}->{tour.id_}): {c2}')
            if c2 > max_c2:
                max_c2 = c2
                rho_best = rho
    return rho_best, max_c2
