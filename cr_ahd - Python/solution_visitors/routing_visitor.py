from abc import ABC, abstractmethod

from profiling import Timer
from utils import opts, InsertionError


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

    @abstractmethod
    def solve_instance(self, instance):
        pass

    @abstractmethod
    def solve_carrier(self, carrier):
        pass

    @abstractmethod
    def find_insertion(self, vertex, carrier):
        pass


class StaticCheapestInsertion(RoutingVisitor):
    def solve_instance(self, instance):
        """
        Use a static construction method to build tours for all carriers via SEQUENTIAL CHEAPEST INSERTION.
        (Can also be used for dynamic route construction if the request-to-carrier assignment is known.)
        """
        assert not instance._solved, f'Instance {instance} has already been solved'
        if self.verbose > 0:
            print(f'STATIC Cheapest Insertion Construction for {instance}:')
        timer = Timer()
        timer.start()

        for c in instance.carriers:
            c.solve(StaticCheapestInsertion(self.verbose, self.plot_level))

        timer.stop()
        self._runtime = timer.duration
        pass

    def solve_carrier(self, carrier):
        # if self.plot_level > 1:
        #     ani = CarrierConstructionAnimation(carrier,
        #                                        f'{instance.id_}{" centralized" if instance.num_carriers == 0 else ""}: '
        #                                        f'Cheapest Insertion construction: {carrier.id_}')

        # TODO why is this necessary here?  why aren't these things computed already before?
        carrier.compute_all_vehicle_cost_and_schedules()

        # construction loop
        for _ in range(len(carrier.unrouted)):
            key, vertex = carrier.unrouted.popitem(last=False)  # sequential removal from list of unrouted from first to last
            vehicle, position, _ = self.find_insertion(vertex, carrier)
            if self.verbose > 0:
                print(f'\tInserting {vertex.id_} into {carrier.id_}.{vehicle.tour.id_}')
            vehicle.tour.insert_and_reset(index=position, vertex=vertex)
            vehicle.tour.compute_cost_and_schedules()

            # if self.plot_level > 1:
            #     ani.add_current_frame()

        assert len(carrier.unrouted) == 0  # just to be completely sure

        # if self.plot_level > 1:
        #     ani.show()
        #     file_name = f'{instance.id_}_{"cen_" if instance.num_carriers == 1 else ""}sta_CI_{carrier.id_ if instance.num_carriers > 1 else ""}.gif'
        #     ani.save(filename=path_output.joinpath('Animations', file_name))
        if self.verbose > 0:
            print(f'Total Route cost of carrier {carrier.id_}: {carrier.cost()}\n')
        pass

    def find_insertion(self, vertex, carrier):
        """
        Checks EVERY vehicle/tour for a feasible insertion and return the cheapest one

        :param vertex: Vertex to be inserted
        :param carrier: carrier which shall insert the vertex into his routes
        :return: triple (vehicle_best, position_best, cost_best) defining the cheapest vehicle/tour, index and
        associated cost to insert the given vertex u
        """
        vehicle_best = None
        cost_best = float('inf')
        position_best = None
        for v in carrier.vehicles:  # check ALL vehicles (also the inactive ones)
            try:
                position, cost = v.tour.find_cheapest_feasible_insertion(u=vertex)
                # TODO tour should not have a find_cheapest_feasible_insertion method! Do i need an InsertionVisitor?
                if self.verbose > 1:
                    print(f'\t\tInsertion cost {vertex.id_} -> {v.id_}: {cost}')
                if cost < cost_best:  # a new cheapest insertion was found -> update the incumbents
                    vehicle_best = v
                    cost_best = cost
                    position_best = position
            except InsertionError:
                if self.verbose > 0:
                    print(f'x\t{vertex.id_} cannot be feasibly inserted into {v.id_}')
        return vehicle_best, position_best, cost_best


class StaticI1Insertion(RoutingVisitor):
    def find_insertion(self, vertex, carrier):
        pass

    def solve_instance(self, instance):
        pass

    def solve_carrier(self, carrier):
        pass


class DynamicInsertion(RoutingVisitor):
    def find_insertion(self, vertex, carrier):
        pass

    def solve_instance(self, instance):
        pass

    def solve_carrier(self, carrier):
        pass


# ========================================================= OLD using strategy
'''
class static_I1_construction(RoutingStrategy):
    def solve(self, instance) -> Tour:
        """
        Use the I1 construction method (Solomon 1987) to build tours for all carriers.
        """
        assert not instance._solved, f'Instance {instance} has already been solved'
        # ensure that tours have been initialized as is done in the original I1 algorithm
        assert instance._initialized, f'Tours must be initialized before solving with I1'

        if self.verbose > 0:
            print(f'STATIC I1 Construction for {instance}:')
        timer = Timer()
        timer.start()

        for c in instance.carriers:
            if self.plot_level > 1:
                ani = CarrierConstructionAnimation(
                    c,
                    f"{instance.id_}{' centralized' if instance.num_carriers == 0 else ''}: Solomon's I1 construction: {c.id_}")

            # TODO why is this necessary here?  why aren't these things computed already before?
            c.compute_all_vehicle_cost_and_schedules(
                instance._distance_matrix)  # tour empty at this point (depot to depot tour)

            # construction loop
            while any(c.unrouted):
                u, vehicle, position, _ = c.find_best_feasible_I1_insertion(instance._distance_matrix)
                if position is not None:  # insert
                    c.unrouted.pop(u.id_)
                    vehicle.tour.insert_and_reset(index=position, vertex=u)
                    vehicle.tour.compute_cost_and_schedules(instance._distance_matrix)
                    if self.verbose > 0:
                        print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                    if self.plot_level > 1:
                        ani.add_current_frame()
                else:
                    if any(c.inactive_vehicles):
                        instance.initialization_visitor
                        c.initialize_tour(c.inactive_vehicles[0], instance._distance_matrix, self._init_method)
                        if self.plot_level > 1:
                            ani.add_current_frame()
                    else:
                        InsertionError('', 'No more vehicles available')

            assert len(c.unrouted) == 0  # just to be on the safe side

            if self.plot_level > 1:
                ani.show()
                ani.save(
                    filename=f'.{path_output}/Animations/{instance.id_}_{"cen_" if instance.num_carriers == 1 else ""}sta_I1_{c.id_ if instance.num_carriers > 1 else ""}.gif')
            if self.verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.cost()}\n')

        timer.stop()
        self._runtime = timer.duration
        return


class dynamic_construction(RoutingStrategy):
    def __init__(self, with_auction: bool = True, verbose=opts['verbose'], plot_level=opts['plot_level']):
        super().__init__(verbose=opts['verbose'], plot_level=opts['plot_level'])
        self._with_auction = with_auction

    def solve(self, instance) -> Tour:
        """
        Iterates over each request, finds the carrier it belongs to (this is necessary because the request-to-carrier
        assignment is pre-determined in the instance files. Optimally, the assignment happens on the fly),
        and then find the optimal insertion position for this request-carrier combination. If with_auction=True,
        the request will be offered in an auction to all carriers. the optimal carrier is determined based on
        cheapest insertion cost.
        """

        assert not instance._solved, f'Instance {instance} has already been solved'

        if self.verbose > 0:
            print(
                f'DYNAMIC Cheapest Insertion Construction {"WITH" if self._with_auction else "WITHOUT"} auction for {instance}:')

        timer = Timer()
        timer.start()
        # find the next request u, that has id number i
        # TODO this can be simplified/made more efficient if the assignment of
        #  a vertex is stored with its class
        #  instance.  In that case, it must also be stored accordingly in the
        #  json file.  Right now, it is not a big
        #  problem since requests are assigned in ascending order, so only the
        #  first request of each carrier must be
        #  checked

        # TODO this function assumes that requests have been assigned to
        #  carriers already which is not really logical
        #  in a real-life case since they arrive dynamically
        for i in range(len(instance.requests)):  # iterate over all requests one by one
            for c in instance.carriers:
                # this loop finds the carrier to which the request is
                # assigned(based on the currently PRE -
                # DETERMINED request-to-carrier assignment).  Optimally, the
                # assignment happens on-the-fly
                try:
                    u_id, u = next(islice(c.unrouted.items(), 1))  # get the first unrouted request of carrier c
                except StopIteration:  # if the next() function cannot return anything due to an exhausted iterator
                    pass
                if int(u_id[1:]) == i:  # if the correct carrier was found, exit the loop
                    break

            if self._with_auction:
                # do the auction
                carrier, vehicle, position, cost = instance.cheapest_insertion_auction(request=u, initial_carrier=c)
                if c != carrier:
                    c.retract_request(u.id_)
                    carrier.assign_request(u)  # assign to auction winner
            else:
                # find cheapest insertion
                carrier = c
                vehicle, position, cost = c.find_cheapest_feasible_insertion(u, instance._distance_matrix)

            # attempt insertion
            try:
                if self.verbose > 0:
                    print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_reset(position, u)
                vehicle.tour.compute_cost_and_schedules(instance._distance_matrix)
                carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
            except TypeError:
                raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

        timer.stop()
        self._runtime = timer.duration
        return
'''