from abc import ABC, abstractmethod

from profiling import Timer
from helper.utils import opts, InsertionError, get_carrier_by_id


class RoutingVisitor(ABC):
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

    @abstractmethod
    def find_insertion(self, carrier, vertex=None):
        pass


class SequentialCheapestInsertion(RoutingVisitor):
    """
    For one customer at a time, will iterate over requests in their order of assignment and finds their cheapest
    insertion based on the route built so far.
    Since this is done one carrier at a time, it will NOT follow the GLOBAL order of request disclosure. However this
    can be easily modified to iterate over the requests on an instance level.
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
                position, cost = vehicle.tour.find_cheapest_feasible_insertion(u=vertex)
                # TODO tour should not have a find_cheapest_feasible_insertion method! Do i need an InsertionVisitor?
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
        assert not instance.solved, f'Instance {instance} has already been solved'
        if self.verbose > 0:
            print(f'STATIC Cheapest Insertion Construction for {instance}:')
        timer = Timer()
        timer.start()

        for carrier in instance.carriers:
            carrier.solve(SequentialCheapestInsertion(self.verbose, self.plot_level))

        timer.stop()
        self._runtime = timer.duration
        instance.solved = True
        pass

    def solve_carrier(self, carrier):
        # if self.plot_level > 1:
        #     ani = CarrierConstructionAnimation(carrier,
        #                     f'{instance.id_}{" centralized" if instance.num_carriers == 0 else ""}: '
        #                     f'Cheapest Insertion construction: {carrier.id_}')

        # TODO why is this necessary here?  why aren't these things computed already before?
        carrier.compute_all_vehicle_cost_and_schedules()

        # construction loop
        while carrier.unrouted:
            vertex = carrier.unrouted[0]
            vehicle, position, _ = self.find_insertion(carrier, vertex)
            vehicle.tour.insert_and_reset(index=position, vertex=vertex)
            vehicle.tour.compute_cost_and_schedules()
            # if self.plot_level > 1:
            #     ani.add_current_frame()

        # if self.plot_level > 1:
        #     ani.show()
        #     file_name = f'{instance.id_}_{"cen_" if instance.num_carriers == 1 else ""}' \
        #                 f'sta_CI_{carrier.id_ if instance.num_carriers > 1 else ""}.gif'
        #     ani.save(filename=path_output.joinpath('Animations', file_name))
        if self.verbose > 0:
            print(f'Total Route cost of carrier {carrier.id_}: {carrier.cost()}\n')
        pass


class StaticI1Insertion(RoutingVisitor):
    def find_insertion(self, carrier, vertex=None):
        """
        Find the next optimal Vertex and its optimal insertion position based on the I1 insertion scheme.
        :return: Tuple(u_best, vehicle_best, rho_best, max_c2)
        """

        vehicle_best = None
        rho_best = None
        u_best = None
        max_c2 = float('-inf')

        for unrouted in carrier.unrouted:  # take the unrouted requests
            for vehicle in carrier.active_vehicles:  # check first the vehicles that are active (to avoid small tours)
                # TODO method must be moved from tour class to (the tour's?) visitor!
                rho, c2 = vehicle.tour.find_best_feasible_I1_insertion(unrouted)
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
        Use the I1 construction method (Solomon 1987) to build tours for all carriers.
        """
        assert not instance.solved, f'Instance {instance} has already been solved'
        # ensure that tours have been initialized as is done in the original I1 algorithm
        assert instance.initialized, f'Tours must be initialized before solving with I1'

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

            carrier.solve(StaticI1Insertion(self.verbose, self.plot_level))

        timer.stop()
        self._runtime = timer.duration
        instance.solved = True
        pass

    def solve_carrier(self, carrier):
        # TODO why is this necessary here?  why aren't these things computed already before?
        carrier.compute_all_vehicle_cost_and_schedules()  # tour empty at this point (depot to depot tour)

        # construction loop
        while carrier.unrouted:
            vertex, vehicle, position, _ = self.find_insertion(carrier)
            if position is not None:  # insert
                vehicle.tour.insert_and_reset(index=position, vertex=vertex)
                vehicle.tour.compute_cost_and_schedules()
                if self.verbose > 0:
                    print(f'\tInserting {vertex.id_} into {carrier.id_}.{vehicle.tour.id_}')
                # if self.plot_level > 1:
                #     ani.add_current_frame()
            else:  # no feasible insertion exists for the vehicles that are active already
                if any(carrier.inactive_vehicles):
                    carrier.initialize_another_tour()
                    # if self.plot_level > 1:
                    #     ani.add_current_frame()
                else:
                    InsertionError('', 'No more vehicles available')

        assert len(carrier.unrouted) == 0  # just to be on the safe side

        # if self.plot_level > 1:
        #     ani.show()
        #     ani.save(
        #         filename=f'.{path_output}/Animations/{instance.id_}_{"cen_" if instance.num_carriers == 1 else ""}' \
        #                  f'sta_I1_{carrier.id_ if instance.num_carriers > 1 else ""}.gif')
        if self.verbose > 0:
            print(f'Total Route cost of carrier {carrier.id_}: {carrier.cost()}\n')
        carrier._solved = True
        pass


class DynamicInsertionWithAuction(RoutingVisitor):
    """Like DynamicInsertion but also has an auction after each request"""

    def find_insertion(self, carrier, vertex):
        """currently uses static cheapest insertion to find the position"""
        vehicle_best, position_best, cost_best = \
            SequentialCheapestInsertion(self.verbose, self.plot_level).find_insertion(carrier, vertex)
        return vehicle_best, position_best, cost_best

    def solve_instance(self, instance):
        assert not instance.solved, f'Instance {instance} has already been solved with {instance.routing_visitor.__class__.__name__}'

        if self.verbose > 0:
            print(f'DYNAMIC Cheapest Insertion Construction WITH AUCTION for {instance}:')

        timer = Timer()
        timer.start()

        # TODO this function assumes that requests have been assigned to
        #  carriers already which is not really logical
        #  in a real-life case since they arrive dynamically
        for carrier in instance.carriers:
            carrier.routing_visitor = SequentialCheapestInsertion()
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
                vehicle.tour.insert_and_reset(position, request)
                vehicle.tour.compute_cost_and_schedules()
            except TypeError:
                raise InsertionError('', f"Cannot insert {request} feasibly into {carrier.id_}.{vehicle.id_}")

        timer.stop()
        self._runtime = timer.duration
        instance.solved = True
        pass

    def solve_carrier(self, carrier):
        raise NotImplementedError
        pass
