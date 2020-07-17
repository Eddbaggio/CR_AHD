from collections import OrderedDict
from copy import deepcopy

import vertex as vx
import vehicle as vh
from tour import Tour
from utils import opts, InsertionError

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Union, OrderedDict as OrderedDictType


class Carrier(object):

    def __init__(self, id_: int, depot: vx.Vertex, vehicles: List[vh.Vehicle],
                 requests: dict = None,
                 unrouted: dict = None):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        if requests is not None:
            self.requests = OrderedDict(
                requests)  # TODO: write comment explaining why these HAVE to be (ordered)dicts instead of lists
        else:
            self.requests = OrderedDict()
        if unrouted is not None:
            self.unrouted = OrderedDict(unrouted)
        else:
            self.unrouted: OrderedDictType[vx.Vertex] = OrderedDict()
        for v in vehicles:
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot])
        pass

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)}, Unrouted:{len(self.unrouted)})'

    def to_dict(self):
        return {
            'id_': self.id_,
            'depot': self.depot.to_dict(),
            'vehicles': [v.to_dict() for v in self.vehicles],
            'requests': [r.to_dict() for r in self.requests.values()]
        }

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

    def initialize_tour(self, vehicle: vh.Vehicle, dist_matrix, earliest_due_date: bool = True):
        assert len(vehicle.tour) == 2, 'Vehicle already has a tour'
        if len(self.unrouted) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed_request(earliest_due_date=True)
            vehicle.tour.insert_and_reset_schedules(index=1, vertex=seed)
            if vehicle.tour.is_feasible(dist_matrix=dist_matrix):
                vehicle.tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                self.unrouted.pop(seed.id_)
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        return

    def static_cheapest_insertion_construction(self, dist_matrix, verbose=opts['verbose'],
                                               plot_level=opts['plot_level']):
        if plot_level > 1:
            fig: plt.Figure = plt.figure()
            plt.grid()
            # ims is a list of lists. each row will contain the list of artists to be drawn at a frame
            ims = []
            full_tour_artists = []
            r_x = [r.coords.x for _, r in self.requests.items()]
            r_y = [r.coords.y for _, r in self.requests.items()]
            requests = plt.plot(r_x, r_y, marker='o', ms=9, mfc='white', mec='black', alpha=0.9, ls='')
            ims.append(requests)

        for v in self.vehicles:
            self.initialize_tour(vehicle=v, dist_matrix=dist_matrix, earliest_due_date=True)
            tour_is_full = False

            if plot_level > 1 and len(self.unrouted) > 0:  # do not plot empty vehicles
                artists = v.tour.plot(color=v.color)
                frame = [*requests, *full_tour_artists, *artists]
                ims.append(frame)

            # add unrouted customers one by one until tour is full
            while len(self.unrouted) > 0 and tour_is_full is False:
                k, u = self.unrouted.popitem(last=False)
                try:
                    # find cheapest feasible insertion position and execute insertion
                    pos, cost = v.tour.cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)
                    if verbose > 0:
                        print(f'\tInserting {u.id_} into {self.id_}.{v.tour.id_}')
                    v.tour.insert_and_reset_schedules(index=pos, vertex=u)
                    v.tour.compute_cost_and_schedules(dist_matrix=dist_matrix)

                    if plot_level > 1:
                        artists = v.tour.plot(color=v.color)
                        frame = [*requests, *full_tour_artists, *artists]
                        ims.append(frame)

                except InsertionError:
                    # re-insert the popped request
                    self.unrouted[k] = u
                    tour_is_full = True
                    if verbose > 0:
                        print(f'x\t{u.id_} cannot be feasibly inserted into {v.tour.id_}')

            if plot_level > 1 and tour_is_full:
                full_tour_artists.extend(artists)

        if plot_level > 1:
            ims.extend([ims[-1]] * 100)
            ani = animation.ArtistAnimation(fig, artists=ims, interval=40, blit=True, repeat=True, repeat_delay=500)
            plt.title(f'Static Cheapest Insertion Construction of {self.id_}')
            plt.show()

        assert len(self.unrouted) == 0, f'{len(self.unrouted)} unrouted customers left for {self}'
        return

    def static_I1_construction(self, dist_matrix, verbose=opts['verbose'], plot_level=opts['plot_level']):
        """Solomon's I1 insertion heuristic from 1987. Following the description of 'Bräysy, Olli; Gendreau,
        Michel (2005): Vehicle Routing Problem with Time Windows, Part I: Route Construction and Local Search
        Algorithms.'
        """

        if True:
            raise DeprecationWarning
            # TODO: check functionality: (1) tours are created sequentially, is that intended? (2) related,
            #  only the current tour is checked for best insertion, not ALL tours as in other approaches
            return

        if plot_level > 1:
            fig: plt.Figure = plt.figure()
            plt.grid()
            # ims is a list of lists. each row will contain the list of artists to be drawn at a frame
            ims = []
            full_tour_artists = []
            r_x = [r.coords.x for _, r in self.requests.items()]
            r_y = [r.coords.y for _, r in self.requests.items()]
            requests = plt.plot(r_x, r_y, marker='o', ms=9, mfc='white', mec='black', alpha=0.9, ls='')
            ims.append(requests)

        for v in self.vehicles:
            # initialize a tour with the first customer with earliest due date
            self.initialize_tour(vehicle=v, dist_matrix=dist_matrix, earliest_due_date=True)
            tour_is_full = False

            if plot_level > 1 and len(self.unrouted) > 0:  # do not plot empty vehicles
                artists = v.tour.plot(color=v.color)
                frame = [*requests, *full_tour_artists, *artists]
                ims.append(frame)

            # fill the tours
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
                        print(f'\tInserting {u_best.id_} into {self.id_}.{v.tour.id_}')

                    v.tour.insert_and_reset_schedules(index=rho_best, vertex=u_best)
                    v.tour.compute_cost_and_schedules(dist_matrix=dist_matrix, ignore_tw=True)
                    self.unrouted.pop(u_best.id_)  # remove u from list of unrouted

                    if plot_level > 1:
                        artists = v.tour.plot(color=v.color)
                        frame = [*requests, *full_tour_artists, *artists]
                        ims.append(frame)

                else:
                    tour_is_full = True

            if plot_level > 1 and tour_is_full:
                full_tour_artists.extend(artists)

        assert len(self.unrouted) == 0, 'Unrouted customers left'

        if plot_level > 1:
            ims.extend([ims[-1]] * 100)
            ani = animation.ArtistAnimation(fig, artists=ims, interval=40, blit=True, repeat=True, repeat_delay=500)
            plt.title(f'Solomon I1 Construction of {self.id_}')
            plt.show()

        return

    def cheapest_feasible_insertion(self, u: vx.Vertex, dist_matrix, verbose=opts['verbose'],
                                    plot_level=opts['plot_level']):
        """
        Checks EVERY vehicle/tour for a feasible insertion and return the cheapest one

        :return: triple (vehicle_best, position_best, cost_best) defining the cheapest vehicle/tour, index and associated cost to insert the given vertex u
        """
        vehicle_best: vh.Vehicle = None
        cost_best = float('inf')
        position_best = None
        for v in self.vehicles:
            try:
                position, cost = v.tour.cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)
                if verbose > 1:
                    print(f'\t\tInsertion cost {u.id_} -> {v.id_}: {cost}')
                if cost < cost_best:
                    vehicle_best = v
                    cost_best = cost
                    position_best = position
            except InsertionError:
                if verbose > 0:
                    print(f'x\t{u.id_} cannot be feasibly inserted into {v.id_}')
        return vehicle_best, position_best, cost_best

    def plot(self, annotate: bool = True, alpha: float = 1):

        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f'tour {self.id_} with cost of {self.route_cost(2)}')

        # plot depot
        ax.plot(*self.depot.coords, marker='s', alpha=alpha, label=self.depot.id_)

        # plot all routes
        for v in self.vehicles:
            if len(v.tour) > 2:
                v.tour.plot(plot_depot=False, annotate=annotate, color=v.color, alpha=alpha)
                ax.legend()
        return
