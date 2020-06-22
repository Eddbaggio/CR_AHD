import vertex as vx
from tour import Tour
from collections import OrderedDict
from utils import opts, InsertionError


class Carrier(object):
    """docstring for Carrier"""

    def __init__(self, id_: int, depot: vx.Vertex, vehicles: list):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        self.requests = OrderedDict()

        for v in vehicles:
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot])
        pass

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)})'

    def assign_request(self, request: vx.Vertex):
        self.requests[request.id_] = request
        # TODO: add dist matrix attribute and extend it with each request assignment? -> inefficient, too much updating? instead have an extra function to extend the matrix whenever necessary?
        pass

    def route_cost(self):
        route_cost = 0
        for v in self.vehicles:
            route_cost += v.tour.cost
        return route_cost

    def find_seed_request(self, earliest_due_date: bool = True) -> vx.Vertex:
        # find request with earliest deadline and initialize pendulum tour
        if earliest_due_date:
            seed = list(self.requests.values())[0]
            for key, request in self.requests.items():
                if request.tw.l < seed.tw.l:
                    seed = self.requests[key]
        return seed

    def _cheapest_insertion_construction(self, dist_matrix, verbose=opts['verbose']):
        """private method, call via construction method of carrier's instance instead"""
        unrouted: OrderedDict = self.requests.copy()
        tours = [v.tour for v in self.vehicles]

        # find request with earliest deadline and initialize pendulum tour
        seed = self.find_seed_request()
        tour: Tour = tours[0]

        print(f'Seed:\n{seed}')
        print(f'Tour:\n{tour}')
        print(f'Distance seed - depot: {dist_matrix.loc[seed.id_, self.depot.id_]}')
        print()

        tour.insert_and_reset_schedules(index=1, vertex=seed)
        if tour.is_feasible(dist_matrix=dist_matrix):
            tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
            unrouted.pop(seed.id_)
        else:
            raise InsertionError('', 'Seed request cannot be inserted feasibly')

        # add unrouted customers one by one
        while len(unrouted) > 0:
            _, u = unrouted.popitem(last=False)  # remove LIFO from unrouted
            inserted = False
            tour_index = 0
            while inserted is False:
                tour = tours[tour_index]
                try:
                    pos, cost = tour.cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)  # will raise error when none is found
                    tour.insert_and_reset_schedules(index=pos, vertex=u)  # add the unrouted element in its cheapest insertion position
                    tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                    inserted = True
                except InsertionError:
                    if verbose > 0:
                        print(f'== {u.id_} cannot be feasibly inserted into {tour.id_}')
                    tour_index += 1
                    pass
        if verbose > 0:
            for v in self.vehicles:
                print()
                print(v.tour)

        pass

        # def I1_construction(self):
        # initialize tour with a seed, a) farthest un-routed request
        # or b) un-routed request with earliest deadline

        # here, b) is implemented

        # def c1(u: vx.Vertex, alpha_1: float, mu):
        #     alpha_2 = 1 - alpha_1
        #     for roh in range(1, len(tour)):
        #         i: vx.Vertex = tour.sequence[roh - 1]
        #         j: vx.Vertex = tour.sequence[roh]

        #         c11 = dist_matrix.loc[i.id_, u.id_] + dist_matrix.loc[u.id_, j.id_] - mu * dist_matrix[i.id_, j.id_]
        # c12 =

        # c1 = alpha_1 * c11 + alpha_2 * c12
