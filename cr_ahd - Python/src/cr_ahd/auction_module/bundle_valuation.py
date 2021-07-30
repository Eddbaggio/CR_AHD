import datetime as dt
import random
from abc import ABC, abstractmethod
from math import ceil
from typing import Sequence, Tuple, List, final

import numpy as np
from scipy.spatial.distance import squareform, pdist

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh, \
    local_search as ls
from src.cr_ahd.utility_module import utils as ut, profiling as pr


def bundle_direct_travel_dist(instance: it.PDPInstance, bundle: Sequence):
    """
    direct travel distances between pickup-delivery pairs of requests in the bundle
    """
    direct_travel_dist = []
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        # NOTE: must use ceil function to ensure triangle inequality
        distance = ceil(instance.distance([pickup], [delivery]))
        direct_travel_dist.append(distance)
    return direct_travel_dist


def bundle_centroid(instance: it.PDPInstance, bundle: Sequence, pd_direct_travel_dist: Sequence):
    """
    centroid of the request’s centers, where the center of request is the midpoint between pickup and delivery
    location. centers are weighted with the length of their request, which is the direct travel distance between
    pickup and delivery of request
    """
    centers = []
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        centers.append(ut.midpoint_(*instance.coords(pickup), *instance.coords(delivery)))

    # there is one instance (run=1+dist=200+rad=150+n=10) for which pickup and delivery of a request are at the same
    # location, in that case, the pd_direct_travel_dist is zero, thus the following if clause is needed
    if sum(pd_direct_travel_dist) == 0:
        weights = None
    else:
        weights = pd_direct_travel_dist

    centroid = ut.Coordinates(*np.average(centers, axis=0, weights=weights))
    return centroid


def bundle_sum_squared_errors(instance: it.PDPInstance, bundle: Sequence[int], centroid: ut.Coordinates):
    """based on cluster analysis evaluation"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    vertex_sse_list = []
    for request in bundle:
        # todo: use the request's midpoint rather than pickup and delivery individually
        for vertex in instance.pickup_delivery_pair(request):
            # todo maybe better use the extended_distance_matrix that includes distances to centroids rather than
            #  computing it all individually
            vertex_sse_list.append(
                ut.euclidean_distance(*instance.coords(vertex), *centroid))
    bundle_sse = sum(vertex_sse_list) / len(vertex_sse_list)
    return bundle_sse


def bundle_cohesion_centroid_based(instance: it.PDPInstance, bundle: Sequence[int], centroid: ut.Coordinates):
    """values closer to 1 are better"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    vertex_cohesion_list = []
    for request in bundle:
        # todo: use the request's midpoint rather than pickup and delivery individually
        for vertex in instance.pickup_delivery_pair(request):
            # todo maybe better use the extended_distance_matrix that includes distances to centroids rather than
            #  computing it all individually
            vertex_cohesion_list.append(
                ut.euclidean_distance(*instance.coords(vertex), *centroid))
    bundle_cohesion = 1 - (sum(vertex_cohesion_list) / len(vertex_cohesion_list))
    return bundle_cohesion


def bundle_separation_centroid_based(centroid: ut.Coordinates, other_centroids: Sequence[ut.Coordinates]):
    """values closer to 1 are better"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    centroid_distances_list = []
    for other_centroid in other_centroids:
        # todo maybe better use the extended_distance_matrix that includes distances to centroids rather than
        #  computing it all individually
        distance = ut.euclidean_distance(*centroid, *other_centroid)
        centroid_distances_list.append(distance)
    bundle_separation = (sum(centroid_distances_list)) / (len(other_centroids))
    return bundle_separation


def bundle_cohesion_graph_based(instance: it.PDPInstance, bundle: Sequence[int]):
    """values closer to 1 are better"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    # since the bundle contains *request* IDs, they need to be unpacked to vertex IDs
    bundle_vertices = []
    for request in bundle:
        bundle_vertices.extend(instance.pickup_delivery_pair(request))

    pair_distances = []
    for i, vertex_i in enumerate(bundle_vertices[:-1]):
        for vertex_j in bundle_vertices[i + 1:]:
            pair_distances.append(ut.euclidean_distance(*instance.coords(vertex_i), *instance.coords(vertex_j)))

    cohesion = 1 - (sum(pair_distances)) / (len(bundle) * (len(bundle) - 1))
    return cohesion


def bundling_extended_distance_matrix(instance: it.PDPInstance, additional_vertex_coords: Sequence[ut.Coordinates]):
    x = instance.x_coords.copy()
    y = instance.y_coords.copy()
    for centroid in additional_vertex_coords:
        x.append(centroid.x)
        y.append(centroid.y)
    # NOTE: must use ceil function to ensure triangle inequality
    return np.ceil(squareform(pdist(np.array(list(zip(x, y)))))).astype('int')


def bundle_vertex_to_centroid_travel_dist(instance: it.PDPInstance,
                                          bundle_idx: int,
                                          bundle: Sequence,
                                          extended_distance_matrix):
    """
    distance of all points in the bundle (pickup and delivery) to the bundle’s centroid. returns these distances in
    tuples of (pickup->centroid, delivery->centroid)

    """
    travel_dist_to_centroid = []
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        # the centroid index for the distance matrix
        centroid_idx = instance.num_depots + 2 * instance.num_requests + bundle_idx
        travel_dist_to_centroid.append(
            (extended_distance_matrix[pickup][centroid_idx], extended_distance_matrix[delivery][centroid_idx]))
    return travel_dist_to_centroid


def bundle_radius(bundle: Sequence, vertex_to_centroid_travel_dist: Sequence[Tuple[float, float]]):
    """average distance of all points in the bundle (pickup and delivery) to the bundle’s centroid."""
    sum_ = 0
    for pickup_dist, delivery_dist in vertex_to_centroid_travel_dist:
        sum_ += pickup_dist + delivery_dist
    return sum_ / (2 * len(vertex_to_centroid_travel_dist))


def bundle_density(direct_travel_dist, vertex_to_centroid_travel_dist: Sequence[Tuple[float, float]]):
    """average direct travel distance of all requests in the bundle, divided by the maximum of the distances of all
    requests to the bundle’s centroid. The distance of a request to the bundle’s centroid is determined by the sum of
    the distances of its pickup point and its delivery point to the centroid.
     """
    avg_direct_travel_dist = sum(direct_travel_dist) / len(direct_travel_dist)
    request_to_centroid_travel_dist = [pickup_dist + delivery_dist for pickup_dist, delivery_dist in
                                       vertex_to_centroid_travel_dist]
    max_dist_to_centroid = max(request_to_centroid_travel_dist)
    return avg_direct_travel_dist / max_dist_to_centroid  # todo causes issues due to python rounding small numbers to zero


def bundle_separation(centroid_a: ut.Coordinates, centroid_b: ut.Coordinates, radius_a: float, radius_b: float):
    """approximates the separation of two bundles a and b."""
    centroid_dist = ut.euclidean_distance(centroid_a.x, centroid_a.y, centroid_b.x, centroid_b.y)
    return centroid_dist / max(radius_a, radius_b)


def bundle_isolation(bundle_centroid: ut.Coordinates,
                     other_bundles_centroid: Sequence[ut.Coordinates],
                     bundle_radius: float,
                     other_bundles_radius: Sequence[float]):
    """minimum separation of bundle from other bundles. returns 0 if there's a single bundle only"""
    if len(other_bundles_centroid) == 0:
        return 0

    min_separation = float('inf')
    for other_centroid, other_radius in zip(other_bundles_centroid, other_bundles_radius):
        separation = bundle_separation(bundle_centroid, other_centroid, bundle_radius, other_radius)
        if separation < min_separation:
            min_separation = separation
    return min_separation


def bundle_total_travel_distance_proxy(instance: it.PDPInstance,
                                       solution: slt.CAHDSolution,
                                       bundle: Sequence[int],
                                       ):
    """
    VERY rough estimate for the total travel distance required to visit all requests in the bundle. Ignores all
    constraints (time windows, vehicle capacity, max tour length, ...)
    """
    routing_sequence = []
    for request in bundle:
        routing_sequence.extend(instance.pickup_delivery_pair(request))
    return instance.distance(routing_sequence[:-1], routing_sequence[1:])


def bundle_total_travel_distance(instance: it.PDPInstance,
                                 solution: slt.CAHDSolution,
                                 bundle: Sequence[int],
                                 ):
    """
    total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic
    ["...using an algorithm proposed by Renaud et al. (2000).]\n
    uses bundle member vertex as depot

    """

    # treat the pickup of the first-to-open delivery vertex as the depot
    min_tw_open = dt.datetime.max
    depot_request = None
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        if solution.tw_open[delivery] < min_tw_open:
            min_tw_open = solution.tw_open[delivery]
            depot_request = request

    # initialize temporary tour with the earliest request
    depot_pickup, depot_delivery = instance.pickup_delivery_pair(depot_request)
    tour_ = tr.Tour('tmp', instance, solution, depot_pickup)
    tour_.insert_and_update(instance, solution, [1], [depot_delivery])

    # insert all remaining requests of the bundle
    tour_construction = cns.MinTravelDistanceInsertion()  # TODO this should be a parameter!
    tour_improvement = mh.PDPVariableNeighborhoodDescent(
        [ls.PDPMove(), ls.PDPTwoOpt()])  # TODO this should be a parameter!
    for request in bundle:
        if request == depot_request:
            continue
        pickup, delivery = instance.pickup_delivery_pair(request)

        # set check_feasibility to False since only the tour length is of interest here
        insertion = tour_construction.best_insertion_for_request_in_tour(instance, solution, tour_, request, False)
        delta, pickup_pos, delivery_pos = insertion

        # if no feasible insertion exists, return inf for the travel distance
        if pickup_pos is None and delivery_pos is None:
            return float('inf')

        # insert, ignores feasibility!
        tour_.insert_and_update(instance, solution, [pickup_pos, delivery_pos], [pickup, delivery])
    tour_improvement.execute_on_tour(instance, solution, tour_)
    return tour_.sum_travel_distance


def los_schulte_similarity(instance: it.PDPInstance, solution: slt.CAHDSolution, vertex1: int, vertex2: int):
    """
    Taken from [1] Los, J., Schulte, F., Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020).
    Decentralized combinatorial auctions for dynamic and large-scale collaborative vehicle routing.

    similarity of two vertices (not requests!) based on a weighted sum of travel time and waiting time
    """

    def waiting_time_seconds(vertex1: int, vertex2: int):
        travel_time = ut.travel_time(instance.distance([vertex1], [vertex2]))
        if solution.tw_open[vertex1] + travel_time > solution.tw_close[vertex2]:
            return float('inf')
        else:
            t0 = max(solution.tw_open[vertex1] + travel_time, solution.tw_open[vertex2])
            t1 = min(solution.tw_close[vertex1] + travel_time, solution.tw_close[vertex2])
            return (t0 - t1).total_seconds()

    # w_ij represents the minimal waiting time (due to time window restrictions) at one of the locations if a vehicle
    # serves both locations immediately after each other
    w_ij = max(0, min(waiting_time_seconds(vertex1, vertex2), waiting_time_seconds(vertex2, vertex1)))
    gamma = 2
    return gamma * ut.travel_time(instance.distance([vertex1], [vertex2])).total_seconds() + w_ij


def bundling_labels_to_bundling(bundling_labels: Sequence[int], auction_request_pool: Sequence[int]):
    """
    duplicate of ut.indices_to_nested_list. uses the indices of bundling_labels to sort the items in
    auction_request_pool into bins.

    Example:
        bundling_labels = [0, 0, 1, 2, 2, 1]
        auction_request_pool = [1, 2, 3, 4, 5, 6]
        bundling = [[1, 2], [3, 6], [4, 5]]

    """
    bundling = [[] for _ in range(max(bundling_labels) + 1)]
    for x, y in zip(auction_request_pool, bundling_labels):
        bundling[y].append(x)
    return bundling


# ======================================================================================================================
# class BundleValuation(ABC):
#     @abstractmethod
#     def evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int]):
#         """evaluates a single bundle"""
#         pass
#
#
# class LosSchulteBundleValuation(BundleValuation):
#     def evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int]):


class BundlingValuation(ABC):
    """
    Class to compute the valuation of a bundling based on some valuation measure(s)
    """

    @final
    def evaluate_bundling_labels(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                 bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """turns bundling labels into bundling and evaluates that bundling"""
        bundling = bundling_labels_to_bundling(bundling_labels, auction_request_pool)
        return self.evaluate_bundling(instance, solution, bundling)

    @abstractmethod
    def evaluate_bundling(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """evaluate a bundling"""
        pass

    def preprocessing(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_request_pool: Sequence[int]):
        pass


class GHProxyBundlingValuation(BundlingValuation):
    @pr.timing
    def evaluate_bundling(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """

        :return: the bundle's valuation/fitness
        """
        centroids = []
        request_direct_travel_distances = []
        num_bundles = len(bundling)

        for bundle in bundling:
            pd_direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
            request_direct_travel_distances.append(pd_direct_travel_dist)
            centroid = bundle_centroid(instance, bundle, pd_direct_travel_dist)
            centroids.append(centroid)

        extended_distance_matrix = bundling_extended_distance_matrix(instance, centroids)

        radii = []
        densities = []
        total_travel_distances = []

        for bundle_idx, bundle in enumerate(bundling):
            # compute the radius
            travel_dist_to_centroid = bundle_vertex_to_centroid_travel_dist(instance, bundle_idx, bundle,
                                                                            extended_distance_matrix)
            radius = bundle_radius(bundle, travel_dist_to_centroid)
            radii.append(radius)

            # compute bundle density
            density = bundle_density(request_direct_travel_distances[bundle_idx], travel_dist_to_centroid)
            densities.append(density)

            # estimating the tour length of the bundle
            approx_travel_dist = bundle_total_travel_distance_proxy(instance, solution, bundle)

            # if there is no feasible tour for this bundle, return a valuation of negative infinity for the whole
            # bundling
            if approx_travel_dist == float('inf'):
                return 0

            total_travel_distances.append(approx_travel_dist)

        isolations = []

        for bundle_idx in range(num_bundles):
            isolation = bundle_isolation(centroids[bundle_idx], centroids[:bundle_idx] + centroids[bundle_idx + 1:],
                                         radii[bundle_idx], radii[:bundle_idx] + radii[bundle_idx + 1:])
            isolations.append(isolation)

        evaluation = (min(isolations) * min(densities)) / (max(total_travel_distances) * num_bundles)
        return evaluation


class MinDistanceBundlingValuation(BundlingValuation):
    """
    The value of a BUNDLING is determined by the minimum over the travel distances per BUNDLE.
    The travel distance of a bundle is determined by building a route that traverses all the bundle's requests using
    the dynamic insertion procedure.
    """

    @pr.timing
    def evaluate_bundling(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        bundle_valuations = []
        for bundle in bundling:
            valuation = bundle_total_travel_distance(instance, solution, bundle)
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class LosSchulteBundlingValuation(BundlingValuation):
    def evaluate_bundling(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """
        uses the similarity measure by Los et al. (2020) to compute a clustering evaluation measure (cohesion)
        """
        bundle_valuations = []
        for bundle in bundling:

            # single-request bundles
            if len(bundle) == 1:
                p0, d0 = instance.pickup_delivery_pair(bundle[0])
                # adjust vertex indices to account for depots
                p0 -= instance.num_depots
                d0 -= instance.num_depots
                # todo: cluster weights acc. to cluster size?
                bundle_valuations.append(self.vertex_similarity_matrix[p0][d0])
                continue

            # lower relatedness values are better
            request_relatedness_list = []

            for i, request0 in enumerate(bundle[:-1]):
                p0, d0 = instance.pickup_delivery_pair(request0)
                # adjust vertex indices to account for depots
                p0 -= instance.num_depots
                d0 -= instance.num_depots

                for j, request1 in enumerate(bundle[i + 1:]):
                    p1, d1 = instance.pickup_delivery_pair(request1)
                    # adjust vertex indices to account for depots
                    p1 -= instance.num_depots
                    d1 -= instance.num_depots

                    # compute relatedness between requests 0 and 1 acc. to the paper's formula (1)
                    relatedness = min(
                        self.vertex_similarity_matrix[p0][d1],
                        self.vertex_similarity_matrix[d0][p1],
                        0.5 * (self.vertex_similarity_matrix[p0][p1] + self.vertex_similarity_matrix[d0][d1])
                    )

                    request_relatedness_list.append(relatedness)

            # collect mean relatedness of the requests in the bundle; lower values are better
            # TODO try max, min or other measures instead of mean?
            # TODO: cluster weights acc. to cluster size?
            bundle_valuations.append(sum(request_relatedness_list) / len(bundle))

        # compute mean valuation of the bundles in the bundling; lower values are better
        # TODO try max, min or other measures instead of mean?
        bundling_valuation = sum(bundle_valuations) / len(bundle_valuations)

        # return inverse, since low values are better and the caller maximizes
        return 1 / bundling_valuation

    def preprocessing(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_request_pool: Sequence[int]):
        """
        pre-compute the pairwise relatedness matrix for all vertices
        """

        n = instance.num_requests

        self.vertex_similarity_matrix = [[0.0] * n * 2 for _ in range(n * 2)]

        for i, request1 in enumerate(instance.requests):
            pickup1, delivery1 = instance.pickup_delivery_pair(request1)
            for j, request2 in enumerate(instance.requests):
                pickup2, delivery2 = instance.pickup_delivery_pair(request2)

                # [1] pickup1 <> delivery1
                self.vertex_similarity_matrix[i][i + n] = los_schulte_similarity(instance, solution, pickup1, delivery1)

                # [2] pickup1 <> pickup2
                self.vertex_similarity_matrix[i][j] = los_schulte_similarity(instance, solution, pickup1, pickup2)

                # [3] pickup1 <> delivery2
                self.vertex_similarity_matrix[i][j + n] = los_schulte_similarity(instance, solution, pickup1, delivery2)

                # [4] delivery1 <> pickup2
                self.vertex_similarity_matrix[i + n][j] = los_schulte_similarity(instance, solution, delivery1, pickup2)

                # [5] delivery1 <> delivery2
                self.vertex_similarity_matrix[i + n][j + n] = los_schulte_similarity(instance, solution, delivery1,
                                                                                     delivery2)

                # [6] pickup2 <> delivery2
                self.vertex_similarity_matrix[j][j + n] = los_schulte_similarity(instance, solution, pickup2, delivery2)


class RandomBundlingValuation(BundlingValuation):
    def evaluate_bundling(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        return random.random()
