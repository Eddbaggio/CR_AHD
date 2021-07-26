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


def _abs_num_requests(carrier_: slt.AHDSolution, num_submitted_requests) -> int:
    """
    returns the absolute number of requests that a carrier shall submit, depending on whether it was initially
    given as an absolute int or a float (relative)
    """
    if isinstance(num_submitted_requests, int):
        return num_submitted_requests
    elif isinstance(num_submitted_requests, float):
        assert num_submitted_requests <= 1, 'If providing a float, must be <=1 to be converted to percentage'
        return round(len(carrier_.assigned_requests) * num_submitted_requests)


class RequestSelectionBehavior(ABC):
    def __init__(self, num_submitted_requests: int):
        self.num_submitted_requests = num_submitted_requests

    @abstractmethod
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        pass

    @staticmethod
    def remove_request_from_carrier(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        """remove the request from the carrier. Finds the tour in which the request is currently served, removes its
        pickup and delivery nodes and reconnects the tour. updates the solution accordingly"""

        carrier_ = solution.carriers[carrier]

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

        :return: the auction_request_pool as a list of request indices and a default
        bundling, i.e. a list of the carrier indices that maps the auction_request_pool to their original carrier.
        """
        auction_request_pool = []
        original_bundling_labels = []
        for carrier in range(instance.num_carriers):
            k = _abs_num_requests(solution.carriers[carrier], self.num_submitted_requests)
            valuations = self._evaluate_requests(instance, solution, carrier)
            # pick the WORST k evaluated requests (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, solution.carriers[carrier].accepted_requests))][:k]

            for request in selected:
                self.remove_request_from_carrier(instance, solution, carrier, request)
                auction_request_pool.append(request)
                original_bundling_labels.append(carrier)

        return auction_request_pool, original_bundling_labels

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
        return np.random.random(len(solution.carriers[carrier].accepted_requests))


class HighestInsertionCostDistance(RequestSelectionBehaviorIndividual):
    """given the current set of routes, returns the n unrouted requests that have the HIGHEST Insertion distance cost.
     NOTE: since routes may not be final, the current highest-marginal-cost request might not have been chosen at an
     earlier or later stage!"""

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        evaluation = []
        carrier_ = solution.carriers[carrier]
        for request in carrier_.accepted_requests:
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
        for r in solution.carriers[carrier].accepted_requests:
            request_revenue = sum([instance.revenue[v] for v in instance.pickup_delivery_pair(r)])
            revenue.append(request_revenue)
        ins_cost = HighestInsertionCostDistance(self.num_submitted_requests)._evaluate_requests(instance, solution,
                                                                                                carrier)
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

class RequestSelectionBehaviorBundle(RequestSelectionBehavior, ABC):
    """
    Select (vor each carrier) a set of bundles based on their combined evaluation of a given measure (e.g. spatial
    proximity)
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_bundling_labels = []

        for carrier in range(instance.num_carriers):
            carrier_ = solution.carriers[carrier]
            k = _abs_num_requests(carrier_, self.num_submitted_requests)

            # make the bundles that shall be evaluated
            bundles = self._create_bundles(instance, solution, carrier, k)

            # find the best bundles for the given carrier
            best_bundle = None
            best_bundle_valuation = -float('inf')
            for bundles in bundles:
                bundle_valuation = self._evaluate_bundle(instance, solution, carrier, bundles)
                if bundle_valuation > best_bundle_valuation:
                    best_bundle = bundles
                    best_bundle_valuation = bundle_valuation

            # carrier's best bundles: retract requests from their tours and add them to auction pool & original bundling
            for request in best_bundle:
                self.remove_request_from_carrier(instance, solution, carrier, request)

                # update auction pool and original bundling candidate
                auction_request_pool.append(request)
                original_bundling_labels.append(carrier)

        return auction_request_pool, original_bundling_labels

    @abstractmethod
    def _create_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, k: int):
        pass

    @abstractmethod
    def _evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                         bundle: Sequence[int]):
        # TODO It could literally be a bundle_valuation strategy that is executed here
        pass


class SpatialBundle(RequestSelectionBehaviorBundle):

    def _create_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                         bundle: Sequence[int]):
        """
        the sum of travel distances of all pairs of requests in this cluster, where the travel distance of a request
        pair is defined as the distance between their origins (pickup locations) plus the distance between
        their destinations (delivery locations).
        """
        pairs = itertools.combinations(bundle, 2)
        evaluation = 0
        for r0, r1 in pairs:
            # negative value: low distance between pairs means high valuation
            # pickup0 to pickup1 + delivery0 to delivery1
            evaluation -= instance.distance(instance.pickup_delivery_pair(r0), instance.pickup_delivery_pair(r1))
        return evaluation


class TemporalRangeBundle(RequestSelectionBehaviorBundle):
    def _create_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                         bundle: Sequence[int]):
        """
        the min-max range of the delivery time windows of all requests inside the cluster
        """
        bundle_delivery_vertices = [instance.pickup_delivery_pair(request)[1] for request in bundle]

        bundle_tw_open = [solution.tw_open[delivery] for delivery in bundle_delivery_vertices]
        bundle_tw_close = [solution.tw_close[delivery] for delivery in bundle_delivery_vertices]
        min_open: dt.datetime = min(bundle_tw_open)
        max_close: dt.datetime = max(bundle_tw_close)
        # negative value: low temporal range means high valuation
        evaluation = - (max_close - min_open).total_seconds()
        return evaluation


class SpatioTemporalBundle(RequestSelectionBehaviorBundle):
    def _create_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(solution.carriers[carrier].accepted_requests, k)

    def _evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                         bundle: Sequence[int]):
        """a weighted sum of spatial and temporal measures"""
        # spatial
        spatial_evaluation = SpatialBundle(self.num_submitted_requests)._evaluate_bundle(instance, solution, carrier,
                                                                                         bundle)
        # normalize to range [0, 1]
        max_pickup_delivery_dist = 0
        min_pickup_delivery_dist = float('inf')
        for i, request1 in enumerate(solution.carriers[carrier].accepted_requests[:-1]):
            for request2 in solution.carriers[carrier].accepted_requests[i + 1:]:
                d = instance.distance(instance.pickup_delivery_pair(request1), instance.pickup_delivery_pair(request2))
                if d > max_pickup_delivery_dist:
                    max_pickup_delivery_dist = d
                if d < min_pickup_delivery_dist:
                    min_pickup_delivery_dist = d

        min_spatial = comb(len(bundle), 2) * (-min_pickup_delivery_dist)
        max_spatial = comb(len(bundle), 2) * (-max_pickup_delivery_dist)
        spatial_evaluation = -(spatial_evaluation - min_spatial) / (max_spatial - min_spatial)

        # temporal range
        temporal_evaluation = TemporalRangeBundle(self.num_submitted_requests)._evaluate_bundle(instance, solution,
                                                                                                carrier, bundle)
        # normalize to range [0, 1]
        min_temporal_range = ut.TW_LENGTH.total_seconds()
        max_temporal_range = (ut.TIME_HORIZON.close - ut.TIME_HORIZON.open).total_seconds()

        temporal_evaluation = -(temporal_evaluation - len(bundle) * (-min_temporal_range)) / (
                len(bundle) * (-max_temporal_range) - len(bundle) * (-min_temporal_range))

        return 0.5 * spatial_evaluation + 0.5 * temporal_evaluation

# class TimeShiftCluster(RequestSelectionBehaviorCluster):
#     """Selects the cluster that yields the highest temporal flexibility when removed"""
#     def _create_bundles(self, instance, solution, carrier, k):
#         """
#         create all possible bundles of size k
#         """
#         return itertools.combinations(solution.carriers[carrier].accepted_requests, k)
