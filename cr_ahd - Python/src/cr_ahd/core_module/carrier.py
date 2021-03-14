from typing import List, Iterable

import matplotlib.pyplot as plt

import src.cr_ahd.core_module.vehicle as vh
import src.cr_ahd.core_module.vertex as vx
from src.cr_ahd.core_module.optimizable import Optimizable
from src.cr_ahd.core_module.tour import Tour


class Carrier(Optimizable):
    def __init__(self,
                 id_: str,
                 depot: vx.DepotVertex,
                 vehicles: List[vh.Vehicle],
                 requests=None,
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

        self._requests = []
        self._unrouted = []
        if requests is None:
            requests = []
        self.assign_requests(requests)
        self._distance_matrix = dist_matrix  # TODO is is meaningful to compute the dist_matrix from requests and depot?

        # self._initializing_visitor: InitializingVisitor = None
        # self._initialized = False
        # self._routing_visitor: RoutingVisitor = None
        # self._solved = False
        # self._finalizing_visitor: FinalizingVisitor = None
        # self._finalized = False
        for v in vehicles:
            v.tour = Tour(id_=v.id_, depot=depot, distance_matrix=dist_matrix)

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot.id_}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)}, Unrouted:{len(self.unrouted_requests)})'

    def cost(self, ndigits: int = 15):
        route_cost = 0
        for v in self.vehicles:
            route_cost += v.tour.cost
        return round(route_cost, ndigits=ndigits)

    @property
    def requests(self):
        # return immutable tuple rather than mutable list
        return tuple(self._requests)

    @property
    def unrouted_requests(self):
        # return immutable tuple rather than list
        return tuple(r for r in self.requests if not r.routed)

    @property
    def routed_requests(self):
        # return immutable tuple rather than list
        return tuple(r for r in self.requests if r.routed)

    @property
    def depot(self):
        return self._depot

    @depot.setter
    def depot(self, depot: vx.DepotVertex):
        depot.carrier_assignment = self.id_
        self._depot = depot
        pass

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

    '''
    @property
    def initializing_visitor(self):
        return self._initializing_visitor

    @initializing_visitor.setter
    def initializing_visitor(self, visitor):
        assert self._initializing_visitor is None or self._initializing_visitor == visitor, f'Initialization visitor already set: '
        self._initializing_visitor = visitor

    @property
    def routing_visitor(self):
        return self._routing_visitor

    @routing_visitor.setter
    def routing_visitor(self, visitor):
        assert self._routing_visitor is None or self._routing_visitor == visitor, f'routing visitor already set'
        self._routing_visitor = visitor

    @property
    def finalizing_visitor(self):
        """the finalizer local search optimization, such as 2opt or 3opt"""
        return self._finalizing_visitor

    @finalizing_visitor.setter
    def finalizing_visitor(self, visitor):
        """Setter for the local search algorithm that can be used to finalize the results"""
        # assert (
        #     not self._finalized), f"carrier has been finalized with visitor {self._finalizing_visitor.__class__.__name__} already!"
        self._finalizing_visitor = visitor
    '''
    def to_dict(self):
        """Nested dictionary representation of self"""
        return {
            'id_': self.id_,
            'depot': self.depot.to_dict(),
            'vehicles': [v.to_dict() for v in self.vehicles],
            # 'initialization_visitor': self.initializing_visitor.__class__.__name__,
            # 'routing_visitor': self.routing_visitor.__class__.__name__,
            # 'finalizing_visitor': self.finalizing_visitor.__class__.__name__,

        }

    def assign_requests(self, requests: List[vx.BaseVertex]):
        """
        Assign a request to the carrier (self). It will be stored in self.requests by its id

        :param requests: the request vertex to be assigned
        :return:
        """
        for r in requests:
            self._requests.append(r)
            self._unrouted.append(r)
            r.carrier_assignment = self.id_
            r.assigned = True
        pass

    def retract_requests_and_update_routes(self, requests: Iterable[vx.Vertex]):
        """
        remove the specified requests from self.requests.

        :param requests: the request to be removed
        :return:
        """
        for request in requests:
            if request.routed:
                routed_tour, routed_index = request.routed
                routed_tour.pop_and_update(routed_index)
            self._requests.remove(request)
            self._unrouted.remove(request)
            request._carrier_assignment = None
            request.assigned = False
        return requests

    '''
    def initialize_another_tour(self):
        """initializes another vehicle pendulum tour, using the InitializingVisitor that has been set the first time
         around"""
        self.initializing_visitor.initialize_carrier(self)
    '''

    '''def initialize(self, visitor: InitializingVisitor):
        """apply visitor's initialization procedure to create a pendulum tour"""
        assert not self._initialized
        self.initializing_visitor = visitor
        visitor.initialize_carrier(self)

    def solve(self, visitor: RoutingVisitor):
        """apply visitor's routing procedure to build routes"""
        assert not self._solved
        self._routing_visitor = visitor
        visitor.solve_carrier(self)

    def finalize(self, visitor: FinalizingVisitor):
        """apply visitor's local search procedure to improve the result after the routing itself has been done"""
        assert not self._finalized
        self._finalizing_visitor = visitor
        visitor.finalize_carrier(self)'''


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
