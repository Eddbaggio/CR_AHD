from abc import ABC, abstractmethod
from itertools import islice

from plotting import CarrierConstructionAnimation
from profiling import Timer
from tour import Tour
from utils import opts, path_output, InsertionError

"""Using the 'Strategy Pattern' to implement different routing algorithms."""


class RoutingStrategy(ABC):

    def __init__(self, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """
        Use a static construction method to build tours for all carriers via SEQUENTIAL CHEAPEST INSERTION.
        (Can also be used for dynamic route construction if the request-to-carrier assignment is known.)

        :param verbose:
        :param plot_level:
        :return:
        """
        self.verbose = verbose
        self.plot_level = plot_level
        self._runtime = 0

    @abstractmethod
    def solve(self, instance) -> Tour:
        pass


class static_cheapest_insertion_construction(RoutingStrategy):
    def solve(self, instance) -> Tour:
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
            if self.plot_level > 1:
                ani = CarrierConstructionAnimation(c,
                                                   f'{instance.id_}{" centralized" if instance.num_carriers == 0 else ""}: '
                                                   f'Cheapest Insertion construction: {c.id_}')

            # TODO why is this necessary here?  why aren't these things computed already before?
            c.compute_all_vehicle_cost_and_schedules(instance.dist_matrix)

            # TODO no initialization of vehicle tours is done, while I1 has initialization - is it unfair?

            # construction loop
            for _ in range(len(c.unrouted)):
                key, u = c.unrouted.popitem(last=False)  # sequential removal from list of unrouted from first to last
                vehicle, position, _ = c.find_cheapest_feasible_insertion(u, instance.dist_matrix)

                if self.verbose > 0:
                    print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                vehicle.tour.insert_and_reset(index=position, vertex=u)
                vehicle.tour.compute_cost_and_schedules(instance.dist_matrix)

                if self.plot_level > 1:
                    ani.add_current_frame()

            assert len(c.unrouted) == 0  # just to be completely sure

            if self.plot_level > 1:
                ani.show()
                file_name = f'{instance.id_}_{"cen_" if instance.num_carriers == 1 else ""}sta_CI_{c.id_ if instance.num_carriers > 1 else ""}.gif'
                ani.save(filename=path_output.joinpath('Animations', file_name))
            if self.verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.cost()}\n')

        timer.stop()
        self._runtime = timer.duration
        return


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
                instance.dist_matrix)  # tour empty at this point (depot to depot tour)

            # construction loop
            while any(c.unrouted):
                u, vehicle, position, _ = c.find_best_feasible_I1_insertion(instance.dist_matrix)
                if position is not None:  # insert
                    c.unrouted.pop(u.id_)
                    vehicle.tour.insert_and_reset(index=position, vertex=u)
                    vehicle.tour.compute_cost_and_schedules(instance.dist_matrix)
                    if self.verbose > 0:
                        print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                    if self.plot_level > 1:
                        ani.add_current_frame()
                else:
                    if any(c.inactive_vehicles):
                        instance.initialization_strategy
                        c.initialize_tour(c.inactive_vehicles[0], instance.dist_matrix, self._init_method)
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
                vehicle, position, cost = c.find_cheapest_feasible_insertion(u, instance.dist_matrix)

            # attempt insertion
            try:
                if self.verbose > 0:
                    print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_reset(position, u)
                vehicle.tour.compute_cost_and_schedules(instance.dist_matrix)
                carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
            except TypeError:
                raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

        timer.stop()
        self._runtime = timer.duration
        return
