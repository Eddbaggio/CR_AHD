from typing import List, Iterable
import datetime as dt
import matplotlib.pyplot as plt
import logging

from src.cr_ahd.core_module import vehicle as vh, vertex as vx, request as rq
from src.cr_ahd.core_module.optimizable import Optimizable
from src.cr_ahd.core_module.tour import Tour

logger = logging.getLogger(__name__)


class Carrier(Optimizable):
    def __init__(self,
                 id_: str,
                 depot: vx.DepotVertex,
                 vehicles: List[vh.Vehicle],
                 requests: List[rq.Request] = None,
                 dist_matrix=None):
        """
        Class that represents carriers in the auction-based collaborative transportation network

        :param id_: unique identifier (usually c0, c1, c2, ...)
        :param depot: Vertex that is the depot of the carrier
        :param vehicles: List of vehicles (of class vh.Vehicle) that belong to this carrier
        :param requests: List of requests that belong to this carrier (not yet assigned)
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
        self._distance_matrix = dist_matrix
        for v in vehicles:
            v.tour = Tour(id_=v.id_, depot=depot, distance_matrix=dist_matrix)

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot.id_}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)}, Unrouted:{len(self.unrouted_requests)})'

    def sum_travel_duration(self):
        sum_travel_durations = dt.timedelta()
        for v in self.vehicles:
            sum_travel_durations += v.tour.sum_travel_duration
        return sum_travel_durations

    def sum_travel_distance(self):
        sum_travel_distances = 0
        for v in self.vehicles:
            sum_travel_distances += v.tour.sum_travel_distance
        return sum_travel_distances

    def vertices(self):
        vertices = []
        for r in self.requests:
            vertices.extend(r.vertices)
        return tuple(vertices)

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
        return self.revenue - self.sum_travel_distance()

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
            logger.debug(f'{r.id_} assigned to {self.id_}')
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
        ax.set_title(f'carrier {self.id_} with total distance of {self.sum_travel_distance}')

        # plot depot
        ax.plot(*self.depot.coords, marker='s', alpha=alpha, label=self.depot.id_, ls='')

        # plot all routes
        for v in self.vehicles:
            if len(v.tour) > 2:
                v.tour.plot(plot_depot=False, annotate=annotate, color=v.color, alpha=alpha)
                ax.legend()
        return
