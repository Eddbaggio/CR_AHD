from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType, Tuple

import matplotlib.pyplot as plt

import vehicle as vh
import vertex as vx
from tour import Tour
from utils import opts, InsertionError


class Carrier(object):

    def __init__(self,
                 id_: str,
                 depot: vx.Vertex,
                 vehicles: List[vh.Vehicle],
                 requests: dict = None,
                 unrouted: dict = None):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        self.num_vehicles = len(self.vehicles)
        if requests is not None:
            self.requests = OrderedDict(requests)
            # TODO: write comment explaining why these HAVE to be (ordered)dicts instead of lists
        else:
            self.requests = OrderedDict()
        if unrouted is not None:
            self.unrouted = OrderedDict(unrouted)
        else:
            self.unrouted: OrderedDictType[str, vx.Vertex] = OrderedDict()
        for v in vehicles:
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot])

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)}, Unrouted:{len(self.unrouted)})'

    def cost(self, ndigits: int = 15):
        route_cost = 0
        for v in self.vehicles:
            route_cost += v.tour.cost
        return round(route_cost, ndigits=ndigits)

    @property
    def revenue(self):
        return sum([v.tour.revenue for v in self.vehicles])

    @property
    def profit(self):
        return self.revenue - self.cost

    @property
    def num_act_veh(self) -> int:
        return sum([1 for v in self.active_vehicles])

    @property
    def active_vehicles(self):
        return [v for v in self.vehicles if v.is_active]

    @property
    def inactive_vehicles(self):
        return [v for v in self.vehicles if not v.is_active]

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
        # TODO: add dist matrix attribute and extend it with each request assignment? -> inefficient,
        #  too much updating? instead have an extra function to extend the matrix whenever necessary?
        pass

    def retract_request(self, request_id: str):
        self.unrouted.pop(request_id)  # remove from initial carrier
        retracted = self.requests.pop(request_id)
        return retracted

    def compute_all_vehicle_cost_and_schedules(self, dist_matrix):
        for v in self.vehicles:
            v.tour.compute_cost_and_schedules(dist_matrix)

    def find_seed_request(self, method: str) -> vx.Vertex:
        # find request with earliest deadline and initialize pendulum tour
        assert method in ['earliest_due_date', 'furthest_distance']
        if method == 'earliest_due_date':
            seed = list(self.unrouted.values())[0]
            for key, request in self.unrouted.items():
                if request.tw.l < seed.tw.l:
                    seed = self.unrouted[key]
        elif method == 'furthest_distance':
            raise NotImplementedError()
        return seed

    def initialize_tour(self, vehicle: vh.Vehicle, dist_matrix, method: str):
        """
        :param vehicle: The vehicle to initialize the tour on
        :param dist_matrix:
        :param method: Either of 'earliest_due_date' or 'furthest_distance'

        :return:
        """
        assert method in ['earliest_due_date', 'furthest_distance']
        assert len(vehicle.tour) == 2, 'Vehicle already has a tour'
        if len(self.unrouted) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed_request(method=method)
            vehicle.tour.insert_and_reset_schedules(index=1, vertex=seed)
            if vehicle.tour.is_feasible(dist_matrix=dist_matrix):
                vehicle.tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                self.unrouted.pop(seed.id_)
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        return

    def find_best_feasible_I1_insertion(self, dist_matrix, verbose=opts['verbose'],
                                        ) -> Tuple[vx.Vertex, vh.Vehicle, int, float]:
        """
        :param dist_matrix:
        :param verbose:
        :return: Tuple(u_best, vehicle_best, rho_best, max_c2)
        """

        vehicle_best = None
        rho_best = None
        u_best = None
        max_c2 = float('-inf')

        for _, u in self.unrouted.items():
            for v in self.active_vehicles:
                rho, c2 = v.tour.find_best_feasible_I1_insertion(u, dist_matrix)
                if c2 > max_c2:
                    if verbose > 1:
                        print(f'^ is the new best c2')
                    vehicle_best = v
                    rho_best = rho
                    u_best = u
                    max_c2 = c2

        return u_best, vehicle_best, rho_best, max_c2

    def find_cheapest_feasible_insertion(self, u: vx.Vertex, dist_matrix, verbose=opts['verbose']):
        """
        Checks EVERY vehicle/tour for a feasible insertion and return the cheapest one

        :return: triple (vehicle_best, position_best, cost_best) defining the cheapest vehicle/tour, index and associated cost to insert the given vertex u
        """
        vehicle_best: vh.Vehicle = None
        cost_best = float('inf')
        position_best = None
        for v in self.vehicles:
            try:
                position, cost = v.tour.find_cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)
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
        ax.set_title(f'carrier {self.id_} with cost of {self.cost(2)}')

        # plot depot
        ax.plot(*self.depot.coords, marker='s', alpha=alpha, label=self.depot.id_, ls='')

        # plot all routes
        for v in self.vehicles:
            if len(v.tour) > 2:
                v.tour.plot(plot_depot=False, annotate=annotate, color=v.color, alpha=alpha)
                ax.legend()
        return
