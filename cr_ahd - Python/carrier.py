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
        """
        Class that represents carriers in the auction-based collaborative transportation network

        :param id_: unique identifier (usually c0, c1, c2, ...)
        :param depot: Vertex that is the depot of the carrier
        :param vehicles: List of vehicles (of class vh.Vehicle) that belong to this carrier
        :param requests: List of requests that are assigned to this carrier
        :param unrouted: List of this carrier's requests that are still unrouted (usually all requests upon
        instantiation of the carrier)
        """
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
        self._initialization_strategy = None

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
        return self.revenue - self.cost()

    @property
    def num_act_veh(self) -> int:
        """Number of active vehicles, i.e. vehicles that serve at least one request Vertex"""
        return sum([1 for v in self.active_vehicles])

    @property
    def active_vehicles(self):
        """List of vehicles that serve at least one request Vertex"""
        return [v for v in self.vehicles if v.is_active]

    @property
    def inactive_vehicles(self):
        """List of vehicles that serve no request Vertex but have depot-depot tour only"""
        return [v for v in self.vehicles if not v.is_active]

    @property
    def initialization_strategy(self):
        return self._initialization_strategy

    @initialization_strategy.setter
    def initialization_strategy(self, strategy):
        assert self._initialization_strategy is None, f'Initialization strategy already set'
        self._initialization_strategy = strategy

    def to_dict(self):
        """Nested dictionary representation of self"""
        return {
            'id_': self.id_,
            'depot': self.depot.to_dict(),
            'vehicles': [v.to_dict() for v in self.vehicles],
            'requests': [r.to_dict() for r in self.requests.values()]
        }

    def assign_request(self, request: vx.Vertex):
        """
        Assign a request to the carrier (self). It will be stored in self.requests by its id

        :param request: the request vertex to be assigned
        :return:
        """
        self.requests[request.id_] = request
        self.unrouted[request.id_] = request
        # TODO: add dist matrix attribute and extend it with each request assignment? -> inefficient,
        #  too much updating? instead have an extra function to extend the matrix whenever necessary?
        pass

    def retract_request(self, request_id: str):
        """
        remove the specified request from self.requests.

        :param request_id: the id of the request to be removed
        :return:
        """
        self.unrouted.pop(request_id)  # auction: remove from initial carrier
        retracted = self.requests.pop(request_id)
        return retracted

    def compute_all_vehicle_cost_and_schedules(self, dist_matrix):
        """
        Computes for all vehicles of this carrier the routing costs and the corresponding schedules (sequence,
        arrival time, service start time) See Vehicle.tour.compute_cost_and_schedules
        :param dist_matrix: The Vertex distance matrix to use
        :return:
        """
        for v in self.vehicles:
            v.tour.compute_cost_and_schedules(dist_matrix)

    '''def find_seed_request(self, method: str) -> vx.Vertex:
        """
        Searches for the best request to be used for a tour initialization based on the given method (Either of
        'earliest_due_date' or 'furthest_distance).

        :param method: Method to identify the best seed request. 'earliest_due_date' finds the request with the
        earliest time window opening time. 'furthest_distance' is not yet implemented
        :return: The seed vertex to use for the pendulum tour
        """

        # find request with earliest deadline and initialize pendulum tour
        assert method in ['earliest_due_date', 'furthest_distance']
        if method == 'earliest_due_date':
            seed = list(self.unrouted.values())[0]
            for key, request in self.unrouted.items():
                if request.tw.l < seed.tw.l:
                    seed = self.unrouted[key]
        elif method == 'furthest_distance':
            raise NotImplementedError()
        return seed'''

    '''
    def initialize_tour(self, vehicle: vh.Vehicle, dist_matrix, method: str):
        """
        Builds a pendulum tour for the specified vehicle by finding a so-called "seed" request. the _method_ parameter
        specifies how the seed is determined (Either of 'earliest_due_date' or 'furthest_distance').

        :param vehicle: The vehicle to initialize the tour on
        :param dist_matrix: distance matrix to be used
        :param method: Either of 'earliest_due_date' or 'furthest_distance'
        """
        assert method in ['earliest_due_date', 'furthest_distance']
        assert len(vehicle.tour) == 2, 'Vehicle already has a tour'
        if len(self.unrouted) > 0:
            # find request with earliest deadline and initialize pendulum tour
            seed = self.find_seed_request(method=method)
            vehicle.tour.insert_and_reset(index=1, vertex=seed)
            if vehicle.tour.is_feasible(dist_matrix=dist_matrix):
                vehicle.tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
                self.unrouted.pop(seed.id_)
            else:
                raise InsertionError('', f'Seed request {seed} cannot be inserted feasibly into {vehicle}')
        return'''

    def initialize(self):
        self._initialization_strategy.initialize()

    def find_best_feasible_I1_insertion(self, dist_matrix, verbose=opts['verbose'],
                                        ) -> Tuple[vx.Vertex, vh.Vehicle, int, float]:
        """
        Find the next optimal Vertex and its optimal insertion position based on the I1 insertion scheme.

        :param dist_matrix:
        :param verbose:
        :return: Tuple(u_best, vehicle_best, rho_best, max_c2)
        """

        vehicle_best = None
        rho_best = None
        u_best = None
        max_c2 = float('-inf')

        for _, u in self.unrouted.items():  # take the unrouted requests
            for v in self.active_vehicles:  # check first the vehicles that are active (to avoid small tours)
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

        :param u: Vertex to be inserted
        :param dist_matrix: distance matrix to consider for insertion cost computations
        :param verbose: level of console output
        :return: triple (vehicle_best, position_best, cost_best) defining the cheapest vehicle/tour, index and
        associated cost to insert the given vertex u
        """
        vehicle_best: vh.Vehicle = None
        cost_best = float('inf')
        position_best = None
        for v in self.vehicles:  # check ALL vehicles (also the inactive ones)
            try:
                position, cost = v.tour.find_cheapest_feasible_insertion(u=u, dist_matrix=dist_matrix)
                if verbose > 1:
                    print(f'\t\tInsertion cost {u.id_} -> {v.id_}: {cost}')
                if cost < cost_best:  # a new cheapest insertion was found -> update the incumbents
                    vehicle_best = v
                    cost_best = cost
                    position_best = position
            except InsertionError:
                if verbose > 0:
                    print(f'x\t{u.id_} cannot be feasibly inserted into {v.id_}')
        return vehicle_best, position_best, cost_best

    def two_opt(self, dist_matrix):
        """Applies the 2-Opt local search operator to all vehicles/tours"""
        for v in self.active_vehicles:
            two_opt_tour = v.tour.two_opt(dist_matrix)
            v.tour = two_opt_tour

    def plot(self, annotate: bool = True, alpha: float = 1):
        """
        Create a matplotlib plot of the carrier and its tours.

        :param annotate: whether the vertices are annotated
        :param alpha: transparency of the nodes
        :return: # TODO why do i not return the fig or ax?
        """
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

    def determine_auction_set(self):
        expensive_requests = []
        for r in self.requests:
            r.get_profit()
        # OR
        for v in self.vehicles:
            for r_id in v.tour.sequence:
                r.get_profit()
    # TODO: determine the value/ profit of each request for the carrier based some criteria (a) demand/distance from
    #  depot, (b) insertion cost (c) c1 or c2 value of I1 algorithm?! (d) ...
