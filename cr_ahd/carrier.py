import vertex as vx
from tour import Tour
from collections import OrderedDict
from utils import opts, InsertionError
from copy import deepcopy


class Carrier(object):
    """docstring for Carrier"""

    def __init__(self, id_: int, depot: vx.Vertex, vehicles: list):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        self.requests = OrderedDict()
        self.unrouted = OrderedDict()
        # TODO: maybe have attributes 'unrouted' and 'routed'?! saves me from copying the self.requests e.g. in _cheapest_insertion_construction

        for v in vehicles:
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot])
        pass

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)})'

    def assign_request(self, request: vx.Vertex):
        self.requests[request.id_] = request
        self.unrouted[request.id_] = request
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
            seed = list(self.unrouted.values())[0]
            for key, request in self.unrouted.items():
                if request.tw.l < seed.tw.l:
                    seed = self.unrouted[key]
        return seed

    def _cheapest_insertion_construction(self, dist_matrix, verbose=opts['verbose']):
        """private method, call via construction method of carrier's instance instead"""
        tours = [v.tour for v in self.vehicles]

        # find request with earliest deadline and initialize pendulum tours
        # for tour in tours:
        #     seed = self.find_seed_request(earliest_due_date=True)
        #     tour.insert_and_reset_schedules(index=1, vertex=seed)
        #     if tour.is_feasible(dist_matrix=dist_matrix):
        #         tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
        #         self.unrouted.pop(seed.id_)
        #     else:
        #         raise InsertionError('', 'Seed request cannot be inserted feasibly')

        # add unrouted customers one by one
        while len(self.unrouted) > 0:
            _, u = self.unrouted.popitem(last=False)  # remove LIFO from unrouted
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

    def _I1_construction(self, dist_matrix, verbose=opts['verbose']):
        """Solomon's I1 insertion heuristic from 1987. Following the description of
         'Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms.'
        """
        tours = [v.tour for v in self.vehicles]

        # find request with earliest deadline and initialize pendulum tours
        for tour in tours:
            seed = self.find_seed_request(earliest_due_date=True)
            tour.insert_and_reset_schedules(index=1, vertex=seed)
            if tour.is_feasible(dist_matrix=dist_matrix):
                tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                self.unrouted.pop(seed.id_)
            else:
                raise InsertionError('', 'Seed request cannot be inserted feasibly')

        # handle unrouted customers one by one
        while len(self.unrouted) > 0:  # TODO: This should be a for loop
            _, u = self.unrouted.popitem(last=False)  # remove LIFO from unrouted
            max_c2 = float('inf')

            for tour in tours:
                # test all insertion positions in the current tour to find the best
                for rho in range(1, len(tour)):
                    i: vx.Vertex = tour.sequence[rho - 1]
                    j: vx.Vertex = tour.sequence[rho]

                    # trivial feasibility check
                    if i.tw.e < u.tw.l and u.tw.e < j.tw.l:

                        # proper feasibility check
                        # TODO: check Solomon (1987) for an efficient feasibility check
                        temp_tour = deepcopy(tour)
                        temp_tour.insert_and_reset_schedules(index=rho, vertex=u)
                        if temp_tour.is_feasible(dist_matrix=dist_matrix):

                            # compute c1 and c2 and update their best values
                            c1 = tour.c1(i_index=rho - 1, u=u, j_index=rho, alpha_1=opts['alpha_1'], mu=opts['mu'], dist_matrix=dist_matrix)
                            if verbose > 1:
                                print(f'c1({u.id_}->{tour.id_}): {c1}')

                            c2 = tour.c2(i_index=rho - 1, u=u, j_index=rho, lambda_=opts['lambda'], c1=c1, dist_matrix=dist_matrix)  # TODO: do I give c1 or min_c1 as input here?!
                            if verbose > 1:
                                print(f'c2({u.id_}->{tour.id_}): {c2}')
                            if c2 < max_c2:
                                if verbose > 1:
                                    print(f'^ is the new best c2')
                                max_c2 = c2
                                rho_best = rho
                                u_best = u
                                tour_best = tour

            if max_c2 < float('inf'):
                if verbose > 1:
                    print(f'Inserting {u_best.id_} into {tour_best.id_}\n')
                tour_best.insert_and_reset_schedules(index=rho_best, vertex=u_best)
                tour_best.compute_cost_and_schedules(dist_matrix=dist_matrix, ignore_tw=True)
            else:
                raise InsertionError('', 'No best insertion candidate found')
