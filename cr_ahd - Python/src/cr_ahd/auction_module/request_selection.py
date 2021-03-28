import itertools
import logging
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.cr_ahd.solving_module.tour_construction import find_cheapest_distance_feasible_insertion
from src.cr_ahd.utility_module.utils import InsertionError

logger = logging.getLogger(__name__)


class RequestSelectionBehavior(ABC):
    def execute(self, instance, num_requests):
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return a dict of lists.

        :param num_requests: the number of requests each carrier shall submit. If fractional, will use the corresponding
         percentage of requests assigned to that carrier. If integer, will use the absolut amount of requests. If not
         enough requests are available, will submit as many as are available
        :param instance:
        :return: dict {carrier_A: List[requests], carrier_B: List[requests]}
        """
        submitted_requests = dict()
        for carrier in instance.carriers:
            if not carrier.unrouted_requests:
                submitted_requests[carrier] = []
            else:
                k = self.abs_num_requests(carrier, num_requests)
                submitted_requests[carrier] = self._select_requests(carrier, k)
                carrier.retract_requests_and_update_routes(submitted_requests[carrier])
        # solution.request_selection = self.__class__.__name__
        return submitted_requests

    @abstractmethod
    def _select_requests(self, carrier, num_requests: int) -> List:
        pass

    def abs_num_requests(self, carrier, num_requests) -> int:
        """returns the absolute number of requests that a carrier shall submit, depending on whether it was initially
        given as an absolute int or a float (relative)"""
        if isinstance(num_requests, int):
            return num_requests
        elif isinstance(num_requests, float):
            assert num_requests <= 1, 'If providing a float, must be <=1 to be converted to percentage'
            return round(len(carrier.unrouted_requests) * num_requests)


class Random(RequestSelectionBehavior):
    """
    returns a random selection of unrouted requests
    """

    def _select_requests(self, carrier, num_requests: int) -> List:
        return np.random.choice(carrier.unrouted_requests, num_requests, replace=False)


class HighestInsertionCostDistance(RequestSelectionBehavior):
    """given the current set of routes, returns the n unrouted requests that has the HIGHEST Insertion distance cost.
     NOTE: since routes may not be final, the current highest-marginal-cost request might not have been chosen at an
     earlier or later stage!"""

    def _select_requests(self, carrier, num_requests: int) -> List:
        selected_requests = []
        for unrouted in carrier.unrouted_requests:
            distance_cost = cheapest_insertion_distance(unrouted, carrier)
            selected_requests.append((unrouted, distance_cost))
        selected_requests, distance_insertion_costs = zip(*sorted(selected_requests, key=lambda x: x[1], reverse=True))
        return selected_requests[:num_requests]


class Cluster(RequestSelectionBehavior):
    """
    Based on: Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies for carriers in auction-based
    collaborations.
    Uses geographical information (intra-cluster distances of cluster members) to select requests that are in close
    proximity
    """

    def _select_requests(self, carrier, num_requests: int) -> List:
        candidate_clusters = itertools.combinations(carrier.unrouted_requests, num_requests)
        best_cluster, best_cluster_evaluation = [], float('inf')
        for candidate in candidate_clusters:
            evaluation = self.cluster_evaluation(candidate, carrier)
            if evaluation < best_cluster_evaluation:
                best_cluster, best_cluster_evaluation = candidate, evaluation
        if best_cluster_evaluation < float('inf'):
            return best_cluster  # , best_cluster_evaluation
        else:
            raise ValueError()

    def cluster_evaluation(self, cluster, carrier):
        pairs = itertools.combinations(cluster, 2)
        evaluation = 0
        for r0, r1 in pairs:
            dist_origins = carrier.distance_matrix[carrier.depot.id_][carrier.depot.id_]  # obviously 0
            dist_destinations = carrier.distance_matrix[r0.id_][r1.id_]
            evaluation += dist_origins + dist_destinations
        return evaluation


# ======================================================================================================
# ======================================================================================================

def cheapest_insertion_distance(request, carrier):
    lowest = float('inf')
    for vehicle in carrier.active_vehicles:
        try:
            _, tour_min = find_cheapest_distance_feasible_insertion(vehicle.tour, request)
            lowest = min(lowest, tour_min)
        except InsertionError:
            continue
    if lowest >= float('inf'):
        t = carrier.inactive_vehicles[0].tour
        t.insert_and_update(1, request)
        lowest = t.sum_travel_distance
        t.pop_and_update(1)
    return lowest
