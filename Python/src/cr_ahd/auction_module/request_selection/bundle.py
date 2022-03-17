import datetime as dt
import itertools
from abc import ABC, abstractmethod
from math import comb
from typing import Sequence

from auction_module.request_selection.request_selection import RequestSelectionBehavior, _abs_num_requests
from core_module import instance as it, solution as slt
from utility_module import utils as ut


class RequestSelectionBehaviorBundle(RequestSelectionBehavior, ABC):
    """
    Select (for each carrier) a set of requests based on their *combined* evaluation of a given measure (e.g. spatial
    proximity of the cluster). This idea of bundled evaluation is also from Gansterer & Hartl (2016)
    """

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_partition_labels = []

        for carrier in solution.carriers:
            k = _abs_num_requests(carrier, self.num_submitted_requests)

            # make the bundles that shall be evaluated
            bundles = self._create_bundles(instance, carrier, k)

            # find the bundle with the MAXIMUM valuation for the given carrier
            best_bundle = None
            best_bundle_valuation = -float('inf')
            for bundle in bundles:
                bundle_valuation = self._evaluate_bundle(instance, carrier, bundle)
                if bundle_valuation > best_bundle_valuation:
                    best_bundle = bundle
                    best_bundle_valuation = bundle_valuation

            # carrier's best bundles: retract requests from their tours and add them to auction pool & original partition
            for request in best_bundle:
                solution.free_requests_from_carriers(instance, [request])

                # update auction pool and original partition candidate
                auction_request_pool.append(request)
                original_partition_labels.append(carrier.id_)

        return auction_request_pool, original_partition_labels

    @abstractmethod
    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        pass

    @abstractmethod
    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        # TODO It could literally be a bundle_valuation strategy that is executed here. Not a partition_valuation though
        pass


class SpatialBundleDSum(RequestSelectionBehaviorBundle):
    """
    Gansterer & Hartl (2016) refer to this one as 'cluster'
    """

    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(carrier.accepted_requests, k)

    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        """
        the sum of travel distances of all pairs of requests in this cluster, where the travel distance of a request
        pair is defined as the distance between their origins (pickup locations) plus the distance between
        their destinations (delivery locations).

        In VRP: the distance between a request pair is the sum of their asymmetric distances
        """
        pairs = itertools.combinations(bundle, 2)
        evaluation = 0
        for r0, r1 in pairs:
            # negative value: low distance between pairs means high valuation
            delivery0 = instance.vertex_from_request(r0)
            delivery1 = instance.vertex_from_request(r1)

            evaluation -= instance.travel_distance([delivery0, delivery1], [delivery1, delivery0])
        return evaluation


class SpatialBundleDMax(RequestSelectionBehaviorBundle):
    """
    Gansterer & Hartl (2016) refer to this one as 'cluster'
    """

    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(carrier.accepted_requests, k)

    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        """
        the sum of travel distances of all pairs of requests in this cluster, where the travel distance of a request
        pair is defined as the distance between their origins (pickup locations) plus the distance between
        their destinations (delivery locations).

        VRP: the travel distance of a request pair is defined as the sum of their asymmetric distances
        """
        pairs = itertools.combinations(bundle, 2)
        evaluation = 0
        for r0, r1 in pairs:
            # negative value: low distance between pairs means high valuation
            d0 = instance.vertex_from_request(r0)
            d1 = instance.vertex_from_request(r1)
            evaluation -= max(instance.travel_distance([d0], [d1]), instance.travel_distance([d1], [d0]))
        return evaluation


class TemporalRangeBundle(RequestSelectionBehaviorBundle):
    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(carrier.accepted_requests, k)

    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        """
        the min-max range of the delivery time windows of all requests inside the cluster
        """
        bundle_delivery_vertices = [instance.vertex_from_request(request) for request in bundle]

        bundle_tw_open = [instance.tw_open[delivery] for delivery in bundle_delivery_vertices]
        bundle_tw_close = [instance.tw_close[delivery] for delivery in bundle_delivery_vertices]
        min_open: dt.datetime = min(bundle_tw_open)
        max_close: dt.datetime = max(bundle_tw_close)
        # negative value: low temporal range means high valuation
        evaluation = - (max_close - min_open).total_seconds()
        return evaluation


class SpatioTemporalBundle(RequestSelectionBehaviorBundle):
    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        """
        create all possible bundles of size k
        """
        return itertools.combinations(carrier.accepted_requests, k)

    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        """a weighted sum of spatial and temporal measures"""
        # spatial
        spatial_evaluation = SpatialBundleDSum(self.num_submitted_requests)._evaluate_bundle(instance, carrier, bundle)
        # normalize to range [0, 1]
        max_pickup_delivery_dist = 0
        min_pickup_delivery_dist = float('inf')
        for i, request1 in enumerate(carrier.accepted_requests[:-1]):
            for request2 in carrier.accepted_requests[i + 1:]:
                delivery1 = instance.vertex_from_request(request1)
                delivery2 = instance.vertex_from_request(request2)
                d = instance.travel_distance([delivery1, delivery2], [delivery2, delivery1])
                if d > max_pickup_delivery_dist:
                    max_pickup_delivery_dist = d
                if d < min_pickup_delivery_dist:
                    min_pickup_delivery_dist = d

        min_spatial = comb(len(bundle), 2) * (-min_pickup_delivery_dist)
        max_spatial = comb(len(bundle), 2) * (-max_pickup_delivery_dist)
        spatial_evaluation = -(spatial_evaluation - min_spatial) / (max_spatial - min_spatial)

        # temporal range
        temporal_evaluation = TemporalRangeBundle(self.num_submitted_requests)._evaluate_bundle(instance, carrier,
                                                                                                bundle)
        # normalize to range [0, 1]
        min_temporal_range = ut.TW_LENGTH.total_seconds()
        max_temporal_range = (ut.EXECUTION_TIME_HORIZON.close - ut.EXECUTION_TIME_HORIZON.open).total_seconds()

        temporal_evaluation = -(temporal_evaluation - len(bundle) * (-min_temporal_range)) / (
                len(bundle) * (-max_temporal_range) - len(bundle) * (-min_temporal_range))

        return 0.5 * spatial_evaluation + 0.5 * temporal_evaluation


class LosSchulteBundle(RequestSelectionBehaviorBundle):
    """
    Selects requests based on their combined evaluation of the bundle evaluation measure by [1] Los, J., Schulte, F.,
    Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020). Decentralized combinatorial auctions for
    dynamic and large-scale collaborative vehicle routing.
    """

    def _create_bundles(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, k: int):
        """
        create all possible bundles of size k
        """
        bundles = itertools.combinations(carrier.accepted_requests, k)
        return bundles

    def _evaluate_bundle(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        # selection for the minimum
        bundle_valuation = bv.LosSchultePartitionValuation()
        bundle_valuation.preprocessing(instance, None)
        # must invert since RequestSelectionBehaviorBundle searches for the maximum valuation and request
        return 1 / bundle_valuation.evaluate_bundle(instance, bundle)