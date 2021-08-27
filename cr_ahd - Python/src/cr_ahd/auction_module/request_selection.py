import datetime as dt
import itertools
import logging
import math
import random
from abc import ABC, abstractmethod
from math import comb, sqrt
from typing import Sequence, Tuple

import numpy as np

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh, neighborhoods as ls
from src.cr_ahd.utility_module import utils as ut
from src.cr_ahd.auction_module import bundle_valuation as bv

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


# =====================================================================================================================
# REQUEST SELECTION BASED ON INDIVIDUAL REQUEST EVALUATION
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
            valuations = []
            for request in solution.carriers[carrier].accepted_requests:
                valuation = self._evaluate_request(instance, solution, carrier, request)
                valuations.append(valuation)

            # pick the WORST k evaluated requests (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, solution.carriers[carrier].accepted_requests))][:k]
            selected.sort()

            for request in selected:
                solution.free_requests_from_carriers(instance, [request])
                auction_request_pool.append(request)
                original_bundling_labels.append(carrier)

        return auction_request_pool, original_bundling_labels

    @abstractmethod
    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        """compute the valuation of the given request for the carrier"""
        pass


class Random(RequestSelectionBehaviorIndividual):
    """
    returns a random selection of unrouted requests
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        return random.random()


class MarginalProfit(RequestSelectionBehaviorIndividual):
    """
        "Requests are selected based on their marginal profits. It is calculated by diminishing the revenue of a
    request by the fulfillment cost. The revenue consists of a fixed and a distance-dependent transportation rate for
    traveling from the pickup to the delivery node. The fulfillment costs are composed of fixed stopping costs and
    the marginal travel cost. The latter can easily be determined by calculating the difference in tour lengths for a
    tour that includes, and a tour that excludes this request. Note that for each evaluation the tour-building
    procedure (see Sect. 3.1) has to be called. Thus, this strategy comes with a relatively high computational
    effort. Requests with the lowest marginal profits are submitted to the pool. This strategy is proposed by Berger
    and Bierwirth (2010)."

    NOTE: due to the lack of some functionality (e.g., efficiently identifying all requests that belong to a tour),
    this is not yet functional.
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        raise NotImplementedError
        # find the request's tour:
        # TODO: would be faster if the tour was stored somewhere as a lookup but this is fine for now
        pickup, delivery = instance.pickup_delivery_pair(request)
        for t in solution.carriers[carrier].tours:
            if pickup in t.routing_sequence:
                tour_ = t
                break
        travel_distance_with_request = tour_.sum_travel_distance

        # create a new tour without the request
        tmp_tour_ = tr.Tour('tmp', instance, tour_.routing_sequence[0])
        requests_in_tour_ = None  # TODO need to find all requests in that tour_
        requests_in_tour_.remove(request)

        insertion = cns.MinTravelDistanceInsertion()
        for insert_request in requests_in_tour_:
            delta, pickup_pos, delivery_pos = insertion.best_insertion_for_request_in_tour(instance, solution,
                                                                                           tmp_tour_, insert_request)
            insertion.execute_insertion_in_tour(instance, solution, tmp_tour_, request, pickup_pos, delivery_pos)
        # improvement
        mh.PDPTWVariableNeighborhoodDescent([ls.PDPMove(), ls.PDPTwoOpt()]).execute_on_tour(instance,
                                                                                            solution,
                                                                                            tmp_tour_)
        travel_distance_without_request = tmp_tour_.sum_travel_distance

        return travel_distance_with_request - travel_distance_without_request


class MarginalProfitProxy(RequestSelectionBehaviorIndividual):
    """
    "Requests are selected based on their marginal profits. It is calculated by diminishing the revenue of a
    request by the fulfillment cost. The revenue consists of a fixed and a distance-dependent transportation rate for
    traveling from the pickup to the delivery node. The fulfillment costs are composed of fixed stopping costs and
    the marginal travel cost. The latter can easily be determined by calculating the difference in tour lengths for a
    tour that includes, and a tour that excludes this request. Note that for each evaluation the tour-building
    procedure (see Sect. 3.1) has to be called. Thus, this strategy comes with a relatively high computational
    effort. Requests with the lowest marginal profits are submitted to the pool. This strategy is proposed by Berger
    and Bierwirth (2010)."

    My implementation is not exactly as described, since i do not call the tour_construction twice per request. Instead,
    the tour that excludes the request is simply calculated by the delta in travel distance when removing the request
    from its tour
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        # find the request's tour:
        tour_ = solution.carriers[carrier].tours[solution.request_to_tour_assignment[request]]
        pickup, delivery = instance.pickup_delivery_pair(request)

        pickup_pos = tour_.routing_sequence.index(pickup)
        delivery_pos = tour_.routing_sequence.index(delivery)

        pickup_predecessor = tour_.routing_sequence[pickup_pos - 1]
        pickup_successor = tour_.routing_sequence[pickup_pos + 1]
        delivery_predecessor = tour_.routing_sequence[delivery_pos - 1]
        delivery_successor = tour_.routing_sequence[delivery_pos + 1]

        # marginal fulfillment cost for the request
        travel_distance_delta = instance.distance([pickup_predecessor, pickup], [pickup, pickup_successor]) - \
                                instance.distance([pickup_predecessor], [pickup_successor]) + \
                                instance.distance([delivery_predecessor, delivery], [delivery, delivery_successor]) - \
                                instance.distance([delivery_predecessor], [delivery_successor])

        # return marginal profit = revenue - fulfillment cost
        return instance.revenue[delivery] - travel_distance_delta


class MinDistanceToForeignDepotDMin(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    MINIMUM of the distances (pickup - depot) and (delivery - depot)
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):

        foreign_depots = solution.carrier_depots[:]
        foreign_depots.pop(carrier)
        foreign_depots = ut.flatten(foreign_depots)

        dist_min = float('inf')
        for depot in foreign_depots:
            dist = min((instance.distance([depot], [vertex]) for vertex in instance.pickup_delivery_pair(request)))
            if dist < dist_min:
                dist_min = dist

        return dist_min


class MinDistanceToForeignDepotDSum(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    SUM of the distances (pickup - depot) and (delivery - depot)
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):

        foreign_depots = solution.carrier_depots[:]
        foreign_depots.pop(carrier)
        foreign_depots = ut.flatten(foreign_depots)

        dist_min = float('inf')
        for depot in foreign_depots:
            dist = sum((instance.distance([depot], [vertex]) for vertex in instance.pickup_delivery_pair(request)))
            if dist < dist_min:
                dist_min = dist

        return dist_min


class ComboRaw(RequestSelectionBehaviorIndividual):
    """
    Combo by Gansterer & Hartl (2016)

    This strategy also combines distance-related and profit-related assessments. A carrier a evaluates a request r
    based on three weighted components: (i) distance to the depot oc of one of the collaborating carriers c (see Fig.
    4), (ii) marginal profit (mprofit), and (iii) distance to the carrier’s own depot oa. Distances are calculated by
    summing up the distances between the pickup node pr and the delivery node dr to the depot (oc or oa).
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        # weighting factors
        alpha1 = 1
        alpha2 = 1
        alpha3 = 1

        # [i] distance to the depot of one of the collaborating carriers
        min_dist_to_foreign_depot = MinDistanceToForeignDepotDSum(None)._evaluate_request(instance, solution, carrier,
                                                                                          request)

        # [ii] marginal profit
        marginal_profit = MarginalProfitProxy(None)._evaluate_request(instance, solution, carrier, request)

        # [iii] distance to the carrier's own depot
        pickup, delivery = instance.pickup_delivery_pair(request)
        assert len(solution.carrier_depots[carrier]) == 1
        own_depot = solution.carrier_depots[carrier][0]
        dist_to_own_depot = instance.distance([pickup, delivery], [own_depot, own_depot])

        # weighted sum of non-standardized [i], [ii], [iii]
        valuation = alpha1 * min_dist_to_foreign_depot + alpha2 * marginal_profit - alpha3 * dist_to_own_depot

        return valuation


class ComboStandardized(RequestSelectionBehaviorIndividual):
    """
    Combo by Gansterer & Hartl (2016)

    This strategy also combines distance-related and profit-related assessments. A carrier a evaluates a request r
    based on three weighted components: (i) distance to the depot oc of one of the collaborating carriers c (see Fig.
    4), (ii) marginal profit (mprofit), and (iii) distance to the carrier’s own depot oa. Distances are calculated by
    summing up the distances between the pickup node pr and the delivery node dr to the depot (oc or oa).
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        Overrides method in RequestSelectionBehaviorIndividual because different components of the valuation function
        need to be normalized before summing them up

        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return

        :return: the auction_request_pool as a list of request indices and a default
        bundling, i.e. a list of the carrier indices that maps the auction_request_pool to their original carrier.
        """
        auction_request_pool = []
        original_bundling_labels = []

        # weighting factors for the three components (min_dist_to_foreign_depot, marginal_profit, dist_to_own_depot)
        # of the valuation function
        alpha1 = 1
        alpha2 = 1
        alpha3 = 1

        for carrier in range(instance.num_carriers):
            k = _abs_num_requests(solution.carriers[carrier], self.num_submitted_requests)
            valuations = []
            for request in solution.carriers[carrier].accepted_requests:
                valuation = self._evaluate_request(instance, solution, carrier, request)
                valuations.append(valuation)

            # normalize the valuation components
            standardized_components = []
            for component_series in zip(*valuations):
                mean = sum(component_series) / len(component_series)
                std = sqrt(sum([(x - mean) ** 2 for x in component_series]) / len(component_series))
                standardized_components.append(((x - mean) / std for x in component_series))

            # compute the weighted sum of the standardized components
            valuations = []
            for comp1, comp2, comp3 in zip(*standardized_components):
                weighted_valuation = alpha1 * comp1 + alpha2 * comp2 + alpha3 * comp3
                valuations.append(weighted_valuation)

            # pick the WORST k evaluated requests (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, solution.carriers[carrier].accepted_requests))][:k]
            selected.sort()

            for request in selected:
                solution.free_requests_from_carriers(instance, [request])
                auction_request_pool.append(request)
                original_bundling_labels.append(carrier)

        return auction_request_pool, original_bundling_labels

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int) -> \
            Tuple[float, float, float]:

        # [i] distance to the depot of one of the collaborating carriers
        min_dist_to_foreign_depot = MinDistanceToForeignDepotDSum(None)._evaluate_request(instance, solution, carrier,
                                                                                          request)

        # [ii] marginal profit
        marginal_profit = MarginalProfitProxy(None)._evaluate_request(instance, solution, carrier, request)

        # [iii] distance to the carrier's own depot
        pickup, delivery = instance.pickup_delivery_pair(request)
        assert len(solution.carrier_depots[carrier]) == 1
        own_depot = solution.carrier_depots[carrier][0]
        dist_to_own_depot = instance.distance([pickup, delivery], [own_depot, own_depot])

        return min_dist_to_foreign_depot, marginal_profit, dist_to_own_depot


class PackedTW(RequestSelectionBehaviorIndividual):
    """
    offer requests from TW slots that are closest to their limit. this way carrier increases flexibility rather than
    profitability
    """

    def _evaluate_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        pass


# =====================================================================================================================
# NEIGHBOR-BASED REQUEST SELECTION
# =====================================================================================================================

class RequestSelectionBehaviorNeighbor(RequestSelectionBehavior, ABC):
    """
    Select (for each carrier) a bundle by finding an initial request and then adding num_submitted_requests-1 more
    requests based on some neighborhood criterion
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_bundling_labels = []

        for carrier in range(instance.num_carriers):
            carrier_ = solution.carriers[carrier]
            k = _abs_num_requests(carrier_, self.num_submitted_requests)

            # find an initial request
            initial_request = self._find_initial_request(instance, solution, carrier)

            # find the best k-1 neighboring ones
            neighbors = self._find_neighbors(instance, solution, carrier, initial_request, k - 1)

            best_bundle = [initial_request] + neighbors
            best_bundle.sort()

            # carrier's best bundles: retract requests from their tours and add them to auction pool & original bundling
            for request in best_bundle:
                solution.free_requests_from_carriers(instance, [request])

                # update auction pool and original bundling candidate
                auction_request_pool.append(request)
                original_bundling_labels.append(carrier)

        return auction_request_pool, original_bundling_labels

    @abstractmethod
    def _find_initial_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass

    @staticmethod
    def _find_neighbors(instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, initial_request: int,
                        num_neighbors: int):
        """
        "Any further request s ∈ Ra is selected based on its closeness to r. Closeness is determined by the sum of
        distances between four nodes (pickup nodes pr , ps and delivery nodes dr , ds )."

        :param instance:
        :param solution:
        :param carrier:
        :param initial_request: r
        :param num_neighbors:
        :return:
        """
        init_pickup, init_delivery = instance.pickup_delivery_pair(initial_request)

        neighbor_valuations = []
        for neighbor in solution.carriers[carrier].accepted_requests:
            if neighbor == initial_request:
                neighbor_valuations.append(float('inf'))

            neigh_pickup, neigh_delivery = instance.pickup_delivery_pair(neighbor)
            valuation = instance.distance([init_pickup, init_pickup, init_delivery, init_delivery],
                                          [neigh_pickup, neigh_delivery, neigh_pickup, neigh_delivery])
            neighbor_valuations.append(valuation)

        # sort by valuations
        best_neigh = [neigh for val, neigh in
                      sorted(zip(neighbor_valuations, solution.carriers[carrier].accepted_requests))]

        return best_neigh[:num_neighbors]


class MarginalProfitProxyNeighbor(RequestSelectionBehaviorNeighbor):
    """
    The first request r ∈ Ra is selected using MarginalProfitProxy. Any further request s ∈ Ra is selected based on
    its closeness to r.
    """

    def _find_initial_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        min_marginal_profit = float('inf')
        initial_request = None
        for request in solution.carriers[carrier].accepted_requests:
            marginal_profit = MarginalProfitProxy(1)._evaluate_request(instance, solution, carrier, request)
            if marginal_profit < min_marginal_profit:
                min_marginal_profit = marginal_profit
                initial_request = request

        return initial_request


# =====================================================================================================================
# BUNDLE-BASED REQUEST SELECTION
# =====================================================================================================================

class RequestSelectionBehaviorBundle(RequestSelectionBehavior, ABC):
    """
    Select (for each carrier) a set of requests based on their combined evaluation of a given measure (e.g. spatial
    proximity of the cluster). This idea of bundled evaluation is also from Gansterer & Hartl (2016)
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_bundling_labels = []

        for carrier in range(instance.num_carriers):
            carrier_ = solution.carriers[carrier]
            k = _abs_num_requests(carrier_, self.num_submitted_requests)

            # make the bundles that shall be evaluated
            bundles = self._create_bundles(instance, solution, carrier, k)

            # find the bundle with the MAXIMUM valuation for the given carrier
            best_bundle = None
            best_bundle_valuation = -float('inf')
            for bundle in bundles:
                bundle_valuation = self._evaluate_bundle(instance, solution, carrier, bundle)
                if bundle_valuation > best_bundle_valuation:
                    best_bundle = bundle
                    best_bundle_valuation = bundle_valuation

            # carrier's best bundles: retract requests from their tours and add them to auction pool & original bundling
            for request in best_bundle:
                solution.free_requests_from_carriers(instance, [request])

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
        # TODO It could literally be a bundle_valuation strategy that is executed here. Not a bundlING_valuation though
        pass


class SpatialBundleDSum(RequestSelectionBehaviorBundle):
    """
    Gansterer & Hartl (2016) refer to this one as 'cluster'
    """

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
            # pickup0 -> pickup1 + delivery0 -> delivery1
            evaluation -= instance.distance(instance.pickup_delivery_pair(r0), instance.pickup_delivery_pair(r1))
        return evaluation


class SpatialBundleDMax(RequestSelectionBehaviorBundle):
    """
    Gansterer & Hartl (2016) refer to this one as 'cluster'
    """

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
            # pickup0 -> pickup1 + delivery0 -> delivery1
            p0, d0 = instance.pickup_delivery_pair(r0)
            p1, d1 = instance.pickup_delivery_pair(r1)
            evaluation -= max(instance.distance([p0], [p1]), instance.distance([d0], [d1]))
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
        spatial_evaluation = SpatialBundleDSum(self.num_submitted_requests)._evaluate_bundle(instance, solution,
                                                                                             carrier,
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


class LosSchulteBundle(RequestSelectionBehaviorBundle):
    """

    """

    def _create_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, k: int):
        """
        create all possible bundles of size k
        """
        bundles = itertools.combinations(solution.carriers[carrier].accepted_requests, k)
        return bundles

    def _evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                         bundle: Sequence[int]):
        # must invert again, since RequestSelectionBehaviorBundle searches for the maximum valuation and request
        # selection for the minimum
        bundle_valuation = bv.LosSchulteBundlingValuation()
        bundle_valuation.preprocessing(instance, solution, None)
        return 1 / bundle_valuation.evaluate_bundle(instance, bundle)

# class TimeShiftCluster(RequestSelectionBehaviorCluster):
#     """Selects the cluster that yields the highest temporal flexibility when removed"""
#     def _create_bundles(self, instance, solution, carrier, k):
#         """
#         create all possible bundles of size k
#         """
#         return itertools.combinations(solution.carriers[carrier].accepted_requests, k)
