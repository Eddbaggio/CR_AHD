from collections import OrderedDict
from copy import deepcopy

import vertex as vx
import vehicle as vh
from tour import Tour
from utils import opts, InsertionError

import matplotlib.pyplot as plt


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

    def route_cost(self, ndigits: int = 15):
        route_cost = 0
        for v in self.vehicles:
            route_cost += v.tour.cost
        return round(route_cost, ndigits=ndigits)

    def find_seed_request(self, earliest_due_date: bool = True) -> vx.Vertex:
        # find request with earliest deadline and initialize pendulum tour
        if earliest_due_date:
            seed = list(self.unrouted.values())[0]
            for key, request in self.unrouted.items():
                if request.tw.l < seed.tw.l:
                    seed = self.unrouted[key]
        return seed

    def initialize_tour(self, tour: Tour, dist_matrix, earliest_due_date: bool = True):
        if len(self.unrouted) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed_request(earliest_due_date=True)
            tour.insert_and_reset_schedules(index=1, vertex=seed)
            if tour.is_feasible(dist_matrix=dist_matrix):
                tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                self.unrouted.pop(seed.id_)

                if opts['plot_level'] > 1:
                    tour.plot()

            else:
                raise InsertionError('', 'Seed request cannot be inserted feasibly')
        return tour

    def static_cheapest_insertion_construction(self, dist_matrix, verbose=opts['verbose'],
                                               plot_level=opts['plot_level']):
        """private method, call via construction method of carrier's instance instead"""
        if plot_level > 1:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots()
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

        v: vh.Vehicle
        for v in self.vehicles:
            self.initialize_tour(v.tour, dist_matrix=dist_matrix, earliest_due_date=True)
            tour_is_full = False

            # add unrouted customers one by one until tour is full
            while len(self.unrouted) > 0 and tour_is_full is False:
                k, u = self.unrouted.popitem(last=False)
                try:
                    # find cheapest feasible insertion position and execute insertion
                    pos, cost = v.tour.cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)
                    if verbose > 0:
                        print(f'\tInserting {u.id_} into {v.tour.id_}')
                    v.tour.insert_and_reset_schedules(index=pos, vertex=u)
                    v.tour.compute_cost_and_schedules(dist_matrix=dist_matrix)

                    if plot_level > 1:
                        v.tour.plot(ax=ax, color=v.color)

                except InsertionError:
                    # re-insert the popped request
                    self.unrouted[k] = u
                    tour_is_full = True
                    if verbose > 0:
                        print(f'x\t{u.id_} cannot be feasibly inserted into {v.tour.id_}')
                    if plot_level > 1:
                        fig.show()

        return

    # def dynamic_cheapest_insertion(self, dist_matrix, verbose=opts['verbose']):
    #     assert len(self.unrouted) == 1, 'More than 1 unrouted customers -> invalid for dynamic insertion'

    def static_I1_construction(self, dist_matrix, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """Solomon's I1 insertion heuristic from 1987. Following the description of
         'Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms.'
        """
        for v in self.vehicles:
            # initialize a tour with the first customer with earliest due date
            self.initialize_tour(v.tour, dist_matrix=dist_matrix, earliest_due_date=True)
            tour_is_full = False

            # handle unrouted customers one by one
            while len(self.unrouted) > 0 and tour_is_full is False:
                rho_best = None
                u_best = None
                max_c2 = float('-inf')

                # find the best request and its best insertion position
                for _, u in self.unrouted.items():
                    for rho in range(1, len(v.tour)):
                        i: vx.Vertex = v.tour.sequence[rho - 1]
                        j: vx.Vertex = v.tour.sequence[rho]

                        # trivial feasibility check
                        if i.tw.e < u.tw.l and u.tw.e < j.tw.l:

                            # proper feasibility check
                            # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
                            temp_tour = deepcopy(v.tour)
                            temp_tour.insert_and_reset_schedules(index=rho, vertex=u)
                            if temp_tour.is_feasible(dist_matrix=dist_matrix):

                                # compute c1 and c2 and update their best values
                                c1 = v.tour.c1(i_index=rho - 1, u=u, j_index=rho, alpha_1=opts['alpha_1'],
                                               mu=opts['mu'],
                                               dist_matrix=dist_matrix)
                                if verbose > 1:
                                    print(f'c1({u.id_}->{v.tour.id_}): {c1}')

                                c2 = v.tour.c2(i_index=rho - 1, u=u, j_index=rho, lambda_=opts['lambda'], c1=c1,
                                               dist_matrix=dist_matrix)
                                if verbose > 1:
                                    print(f'c2({u.id_}->{v.tour.id_}): {c2}')
                                if c2 > max_c2:
                                    if verbose > 1:
                                        print(f'^ is the new best c2')
                                    max_c2 = c2
                                    rho_best = rho
                                    u_best = u

                if max_c2 > float('-inf'):
                    if verbose > 0:
                        print(f'\tInserting {u_best.id_} into {v.tour.id_}')
                    v.tour.insert_and_reset_schedules(index=rho_best, vertex=u_best)
                    v.tour.compute_cost_and_schedules(dist_matrix=dist_matrix, ignore_tw=True)
                    self.unrouted.pop(u_best.id_)  # remove u from list of unrouted

                    if plot_level > 1:
                        v.tour.plot(ax=ax, color=v.color)

                else:
                    tour_is_full = True
                    # raise InsertionError('', 'No best insertion candidate found')
        if plot_level > 1:
            plt.show()
        return

    def plot(self, annotate: bool = True, alpha: float = 1):

        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f'tour {self.id_} with cost of {self.route_cost(2)}')

        # plot depot
        ax.plot(*self.depot.coords, marker='s', alpha=alpha, label=self.depot.id_)

        # plot all routes on the same axes
        for v in self.vehicles:
            if len(v.tour) > 2:
                v.tour.plot(ax=ax, plot_depot=False, annotate=annotate, color=v.color, alpha=alpha)
                ax.legend()
        return
