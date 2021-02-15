from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType, Tuple

import matplotlib.pyplot as plt

import vehicle as vh
import vertex as vx
from Optimizable import Optimizable
from solution_visitors.initializing_visitor import InitializingVisitor
from solution_visitors.local_search_visitor import FinalizingVisitor
from solution_visitors.routing_visitor import RoutingVisitor
from tour import Tour
from utils import opts, InsertionError


class Carrier(Optimizable):
    def __init__(self,
                 id_: str,
                 depot: vx.Vertex,
                 vehicles: List[vh.Vehicle],
                 requests: dict = None,
                 unrouted: dict = None,
                 dist_matrix=None):
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
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot], distance_matrix=dist_matrix)
        self._distance_matrix = dist_matrix
        self._initializing_visitor: InitializingVisitor = None
        self._initialized = False
        self._routing_visitor: RoutingVisitor = None
        self._solved = False
        self._finalizing_visitor: FinalizingVisitor = None
        self._finalized = False

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)}, Unrouted:{len(self.unrouted)})'

    def cost(self, ndigits: int = 15):
        route_cost = 0
        for v in self.vehicles:
            route_cost += v.tour.cost
        return round(route_cost, ndigits=ndigits)

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, dist_matrix):
        self._distance_matrix = dist_matrix

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
    def initializing_visitor(self):
        return self._initializing_visitor

    @initializing_visitor.setter
    def initializing_visitor(self, visitor):
        assert self._initializing_visitor is None, f'Initialization visitor already set'
        self._initializing_visitor = visitor

    @property
    def routing_visitor(self):
        return self._routing_visitor

    @routing_visitor.setter
    def routing_visitor(self, visitor):
        assert self._routing_visitor is None, f'routing visitor already set'
        self._routing_visitor = visitor

    @property
    def finalizing_visitor(self):
        """the finalizer local search optimization, such as 2opt or 3opt"""
        return self._finalizing_visitor

    @finalizing_visitor.setter
    def finalizing_visitor(self, visitor):
        """Setter for the local search algorithm that can be used to finalize the results"""
        assert (
            not self._finalized), f"carrier has been finalized with visitor {self._finalizing_visitor.__class__.__name__} already!"
        self._finalizing_visitor = visitor

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
        request.carrier_assignment = self.id_
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
        retracted: vx.Vertex = self.requests.pop(request_id)
        retracted.carrier_assignment = None
        return retracted

    def compute_all_vehicle_cost_and_schedules(self):
        """
        Computes for all vehicles of this carrier the routing costs and the corresponding schedules (sequence,
        arrival time, service start time) See Vehicle.tour.compute_cost_and_schedules
        :return:
        """
        for v in self.vehicles:
            v.tour.compute_cost_and_schedules()

    def initialize(self, visitor: InitializingVisitor):
        """apply visitor's initialization procedure to create a pendulum tour"""
        assert not self._initialized
        self.initializing_visitor = visitor
        visitor.initialize_carrier(self)

    def initialize_another_tour(self):
        """does not require the carrier to be uninitialized. For now, only allows to use the initializingVisitor that
        has been set the first time around"""
        self.initializing_visitor.initialize_carrier(self)

    def solve(self, visitor: RoutingVisitor):
        """apply visitor's routing procedure to build routes"""
        assert not self._solved
        self._routing_visitor = visitor
        visitor.solve_carrier(self)

    def finalize(self, visitor: FinalizingVisitor):
        """apply visitor's local search procedure to improve the result after the routing itself has been done"""
        assert not self._finalized
        self._finalizing_visitor = visitor
        visitor.finalize_carrier(self)

    def reset_solution(self):
        for vehicle in self.vehicles:
            vehicle.tour.reset_solution()

    '''def find_best_feasible_I1_insertion(self, verbose=opts['verbose'],
                                        ) -> Tuple[vx.Vertex, vh.Vehicle, int, float]:
        """
        Find the next optimal Vertex and its optimal insertion position based on the I1 insertion scheme.

        :param dist_matrix:
        :return: Tuple(u_best, vehicle_best, rho_best, max_c2)
        """

        vehicle_best = None
        rho_best = None
        u_best = None
        max_c2 = float('-inf')

        for _, u in self.unrouted.items():  # take the unrouted requests
            for v in self.active_vehicles:  # check first the vehicles that are active (to avoid small tours)
                rho, c2 = v.tour.find_best_feasible_I1_insertion(u)
                if c2 > max_c2:
                    if verbose > 1:
                        print(f'^ is the new best c2')
                    vehicle_best = v
                    rho_best = rho
                    u_best = u
                    max_c2 = c2

        return u_best, vehicle_best, rho_best, max_c2'''

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
