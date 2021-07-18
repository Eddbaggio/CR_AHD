import datetime as dt
import itertools
import logging
from abc import ABC, abstractmethod
from math import comb
from typing import Sequence, Iterable

import numpy as np

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


def _abs_num_requests(carrier_: slt.AHDSolution, num_requests) -> int:
    """
    returns the absolute number of requests that a carrier shall submit, depending on whether it was initially
    given as an absolute int or a float (relative)
    """
    if isinstance(num_requests, int):
        return num_requests
    elif isinstance(num_requests, float):
        assert num_requests <= 1, 'If providing a float, must be <=1 to be converted to percentage'
        return round(len(carrier_.assigned_requests) * num_requests)


class RequestSelectionBehavior(ABC):
    def __init__(self, num_submitted_requests: int):
        self.num_submitted_requests = num_submitted_requests

    @abstractmethod
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        pass


# =====================================================================================================================
# INDIVIDUAL REQUEST EVALUATION
# =====================================================================================================================

class RequestSelectionBehaviorIndividual(RequestSelectionBehavior, ABC):
    """
    select (for each carrier) a set of bundles based on their individual evaluation of some quality criterion
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return

        :param solution:
        :param instance:
        :return: the auction pool as a list of request indices and a default bundling, i.e. a list of the carriers that
        originally submitted the request.
        """
        auction_pool = []
        original_bundles = []
        for carrier in range(instance.num_carriers):
            k = _abs_num_requests(solution.carriers[carrier], self.num_submitted_requests)
            valuations = self._evaluate_requests(instance, solution, carrier)
            # pick the worst k evaluated requests (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, solution.carriers[carrier].unrouted_requests))][:k]

            for s in selected:
                solution.carriers[carrier].assigned_requests.remove(s)
                solution.carriers[carrier].accepted_requests.remove(s)
                solution.request_to_carrier_assignment[s] = np.nan
                solution.unassigned_requests.append(s)

                auction_pool.append(s)
                original_bundles.append(carrier)

        return auction_pool, original_bundles

    @abstractmethod
    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        """
        compute the carrier's valuation for all unrouted requests of that carrier.
        NOTE: a high number corresponds to a high valuation

        """
        pass


class Random(RequestSelectionBehaviorIndividual):
    """
    returns a random selection of unrouted requests
    """

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        return np.random.random(len(solution.carriers[carrier].unrouted_requests))


class HighestInsertionCostDistance(RequestSelectionBehaviorIndividual):
    """given the current set of routes, returns the n unrouted requests that have the HIGHEST Insertion distance cost.
     NOTE: since routes may not be final, the current highest-marginal-cost request might not have been chosen at an
     earlier or later stage!"""

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        evaluation = []
        carrier_ = solution.carriers[carrier]
        for request in carrier_.unrouted_requests:
            best_delta_for_r = float('inf')
            for tour in range(carrier_.num_tours()):
                # cheapest way to fit request into tour
                delta, _, _ = cns.MinTravelDistanceInsertion()._tour_cheapest_dist_insertion(instance, solution,
                                                                                             carrier, tour,
                                                                                             request)
                if delta < best_delta_for_r:
                    best_delta_for_r = delta
            # if no feasible insertion for the current request was found, attempt to create a new tour, if that's not
            # feasible the best_delta_for_r will be infinity
            if best_delta_for_r == float('inf'):
                if carrier_.num_tours() < instance.carriers_max_num_tours:
                    pickup, delivery = instance.pickup_delivery_pair(request)
                    best_delta_for_r = instance.distance([carrier, pickup, delivery], [pickup, delivery, carrier])
            # collect the NEGATIVE value, since high insertion cost mean a low valuation for the carrier
            evaluation.append(-best_delta_for_r)
        return evaluation


class LowestProfit(RequestSelectionBehaviorIndividual):
    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        raise NotImplementedError
        # this is a wrong implementation of mProfit (Gansterer & Hartl / Berger & Bierwirth). It SHOULD be:
        # fulfillment costs = (tour costs with the request) - (tour costs without the request)

        revenue = []
        for r in solution.carriers[carrier].unrouted_requests:
            request_revenue = sum([instance.revenue[v] for v in instance.pickup_delivery_pair(r)])
            revenue.append(request_revenue)
        ins_cost = HighestInsertionCostDistance()._evaluate_requests(instance, solution, carrier)
        # return the profit, high profit means high valuation
        return [rev + cost for rev, cost in zip(revenue, ins_cost)]


class PackedTW(RequestSelectionBehaviorIndividual):
    """
    offer requests from TW slots that are closest to their limit. this way carrier increases flexibility rather than
    profitability
    """
    pass


# =====================================================================================================================
# BUNDLE-BASED EVALUATION
# =====================================================================================================================

class RequestSelectionBehaviorCluster(RequestSelectionBehavior, ABC):
    """
    Select (vor each carrier) a set of bundles based on their combined evaluation of a given measure (e.g. spatial
    proximity)
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        auction_pool = []
        original_bundles = []

        for carrier in range(instance.num_carriers):
            carrier_ = solution.carriers[carrier]
            k = _abs_num_requests(carrier_, self.num_submitted_requests)

            # make the clusters that shall be evaluated
            clusters = self._create_clusters(instance, solution, carrier, k)

            # find the best cluster for the given carrier
            best_cluster = None
            best_cluster_valuation = -float('inf')
            for cluster in clusters:
                bundle_valuation = self._evaluate_cluster(instance, solution, carrier, cluster)
                if bundle_valuation > best_cluster_valuation:
                    best_cluster = cluster
                    best_cluster_valuation = bundle_valuation

            # carrier's best cluster: retract requests from their tours and add them to auction pool & original bundling
            for request in best_cluster:

                # find the request's tour:
                # TODO: would be faster if the tour was stored somewhere but this is fine for now
                pickup, delivery = instance.pickup_delivery_pair(request)
                for t in carrier_.tours:
                    if pickup in t.routing_sequence:
                        tour_ = t
                        break

                # destroy & repair, i.e. remove the request from it's tour
                pickup_pos = tour_.routing_sequence.index(pickup)
                delivery_pos = tour_.routing_sequence.index(delivery)
                tour_.pop_and_update(instance, solution, [pickup_pos, delivery_pos])

                # retract the request from the carrier
                carrier_.assigned_requests.remove(request)
                carrier_.accepted_requests.remove(request)
                carrier_.routed_requests.remove(request)
                solution.request_to_carrier_assignment[request] = np.nan
                solution.unassigned_requests.append(request)

                # update auction pool and original bundling candidate
                auction_pool.append(request)
                original_bundles.append(carrier)

        return auction_pool, original_bundles

    @abstractmethod
    def _create_clusters(self, instance, solution, carrier, k):
        pass

    @abstractmethod
    def _evaluate_cluster(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, cluster):
        pass


class SpatialCluster(RequestSelectionBehaviorCluster):

    def _create_clusters(self, instance, solution, carrier, k):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_cluster(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, cluster):
        """
        the sum of travel distances of all pairs of requests in this cluster, where the travel distance of a request
        pair is defined as the distance between their origins (pickup locations) plus the distance between
        their destinations (delivery locations).
        """
        pairs = itertools.combinations(cluster, 2)
        evaluation = 0
        for r0, r1 in pairs:
            # negative value: low distance between pairs means high valuation
            # pickup0 to pickup1 + delivery0 to delivery1
            evaluation -= instance.distance(instance.pickup_delivery_pair(r0), instance.pickup_delivery_pair(r1))
        return evaluation


class TemporalRangeCluster(RequestSelectionBehaviorCluster):
    def _create_clusters(self, instance, solution, carrier, k):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_cluster(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, cluster):
        """
        the min-max range of the delivery time windows of all requests inside the cluster
        """
        cluster_delivery_vertices = [instance.pickup_delivery_pair(request)[1] for request in cluster]

        cluster_tw_open = [solution.tw_open[delivery] for delivery in cluster_delivery_vertices]
        cluster_tw_close = [solution.tw_close[delivery] for delivery in cluster_delivery_vertices]
        min_open: dt.datetime = min(cluster_tw_open)
        max_close: dt.datetime = max(cluster_tw_close)
        # negative value: low temporal range means high valuation
        evaluation = - (max_close - min_open).total_seconds()
        return evaluation


class SpatioTemporalCluster(RequestSelectionBehaviorCluster):
    def _create_clusters(self, instance, solution, carrier, k):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_cluster(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, cluster: Sequence):
        """a weighted sum of spatial and temporal measures"""
        # spatial
        spatial_evaluation = SpatialCluster(self.num_submitted_requests)._evaluate_cluster(instance, solution, carrier,
                                                                                           cluster)
        # normalize to range [0, 1]
        max_pickup_delivery_dist = 0
        min_pickup_delivery_dist = float('inf')
        for request1 in solution.carriers[carrier].accepted_requests[:-1]:
            for request2 in solution.carriers[carrier].accepted_requests[1:]:
                d = instance.distance(instance.pickup_delivery_pair(request1), instance.pickup_delivery_pair(request2))
                if d > max_pickup_delivery_dist:
                    max_pickup_delivery_dist = d
                if d < min_pickup_delivery_dist:
                    min_pickup_delivery_dist = d

        min_spatial = comb(len(cluster), 2) * (-min_pickup_delivery_dist)
        max_spatial = comb(len(cluster), 2) * (-max_pickup_delivery_dist)
        spatial_evaluation = -(spatial_evaluation - min_spatial) / (max_spatial - min_spatial)

        # temporal range
        temporal_evaluation = TemporalRangeCluster(self.num_submitted_requests)._evaluate_cluster(instance, solution,
                                                                                                  carrier, cluster)
        # normalize to range [0, 1]
        min_temporal_range = ut.TW_LENGTH.total_seconds()
        max_temporal_range = (ut.TIME_HORIZON.close - ut.TIME_HORIZON.open).total_seconds()

        temporal_evaluation = -(temporal_evaluation - len(cluster) * (-min_temporal_range)) / (
                len(cluster) * (-max_temporal_range) - len(cluster) * (-min_temporal_range))

        return 0.5 * spatial_evaluation + 0.5 * temporal_evaluation

# class TimeShiftCluster(RequestSelectionBehaviorCluster):
#     """Selects the cluster that yields the highest temporal flexibility when removed"""
#     def _create_clusters(self, instance, solution, carrier, k):
#         """
#         create all possible bundles of size k
#         """
#         return itertools.combinations(solution.carriers[carrier].accepted_requests, k)
