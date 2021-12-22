import datetime as dt
import random
from abc import ABC, abstractmethod
from math import ceil
from typing import Sequence, Tuple, List, final

import numpy as np
from scipy.spatial.distance import squareform, pdist

from core_module import instance as it, solution as slt, tour as tr
from routing_module import tour_construction as cns, metaheuristics as mh, \
    neighborhoods as nh
from utility_module import utils as ut


# ======================================================================================================================
# STAND-ALONE BUNDLE EVALUATION MEASURES
# ======================================================================================================================


def bundle_direct_travel_dist(instance: it.MDPDPTWInstance, bundle: Sequence):
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


def bundle_centroid(instance: it.MDPDPTWInstance, bundle: Sequence, pd_direct_travel_dist: Sequence):
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


def bundle_sum_squared_errors(instance: it.MDPDPTWInstance, bundle: Sequence[int], centroid: ut.Coordinates):
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


def bundle_cohesion_centroid_based(instance: it.MDPDPTWInstance, bundle: Sequence[int], centroid: ut.Coordinates):
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


def bundle_cohesion_graph_based(instance: it.MDPDPTWInstance, bundle: Sequence[int]):
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


def bundling_extended_distance_matrix(instance: it.MDPDPTWInstance, additional_vertex_coords: Sequence[ut.Coordinates]):
    x = instance.vertex_x_coords.copy()
    y = instance.vertex_y_coords.copy()
    for centroid in additional_vertex_coords:
        x.append(centroid.x)
        y.append(centroid.y)
    # NOTE: must use ceil function to ensure triangle inequality
    return np.ceil(squareform(pdist(np.array(list(zip(x, y)))))).astype('int')


def bundle_vertex_to_centroid_travel_dist(instance: it.MDPDPTWInstance,
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
        centroid_idx = instance.num_carriers + 2 * instance.num_requests + bundle_idx
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


def bundle_total_travel_distance_proxy(instance: it.MDPDPTWInstance, bundle: Sequence[int]):
    """
    VERY rough estimate for the total travel distance required to visit all requests in the bundle. Ignores all
    constraints (time windows, vehicle capacity, max tour length, ...)
    """
    routing_sequence = []
    for request in bundle:
        routing_sequence.extend(instance.pickup_delivery_pair(request))
    return instance.distance(routing_sequence[:-1], routing_sequence[1:])


def bundle_total_travel_distance(instance: it.MDPDPTWInstance,
                                 solution: slt.CAHDSolution,
                                 bundle: Sequence[int],
                                 ):
    """
    total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic
    ["...using an algorithm proposed by Renaud et al. (2000).]\n
    uses bundle member vertex as depot. Does not check feasibility of the tour

    """

    # treat the pickup of the first-to-open delivery vertex as the depot
    min_tw_open = dt.datetime.max
    depot_request = None
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        if instance.tw_open[delivery] < min_tw_open:
            min_tw_open = instance.tw_open[delivery]
            depot_request = request

    # initialize temporary tour with the earliest request
    depot_pickup, depot_delivery = instance.pickup_delivery_pair(depot_request)
    tmp_tour_ = tr.Tour('tmp', depot_pickup)
    tmp_tour_.insert_and_update(instance, [1], [depot_delivery])

    # insert all remaining requests of the bundle
    tour_construction = cns.MinTravelDistanceInsertion()  # TODO this should be a parameter!
    tour_improvement = mh.PDPTWVariableNeighborhoodDescent(
        [nh.PDPMove(), nh.PDPTwoOpt()])  # TODO this should be a parameter!
    for request in bundle:
        if request == depot_request:
            continue
        pickup, delivery = instance.pickup_delivery_pair(request)

        # set check_feasibility to False since only the tour length is of interest here
        insertion = tour_construction.best_insertion_for_request_in_tour(instance, tmp_tour_, request, False)
        delta, pickup_pos, delivery_pos = insertion

        # if no feasible insertion exists, return inf for the travel distance
        if pickup_pos is None and delivery_pos is None:
            return float('inf')

        # insert, ignores feasibility!
        tmp_tour_.insert_and_update(instance, [pickup_pos, delivery_pos], [pickup, delivery])
    tour_improvement.execute_on_tour(instance, tmp_tour_)
    return tmp_tour_.sum_travel_distance


def los_schulte_vertex_similarity(instance: it.MDPDPTWInstance, vertex1: int, vertex2: int):
    """
    Following [1] Los, J., Schulte, F., Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020).
    Decentralized combinatorial auctions for dynamic and large-scale collaborative vehicle routing.

    similarity of two vertices (not requests!) based on a weighted sum of travel time and waiting time
    """

    def waiting_time_seconds(vertex1: int, vertex2: int):
        travel_time = instance.travel_duration([vertex1], [vertex2])
        if instance.tw_open[vertex1] + travel_time > instance.tw_close[vertex2]:
            return float('inf')
        else:
            t0 = max(instance.tw_open[vertex1] + travel_time, instance.tw_open[vertex2])
            t1 = min(instance.tw_close[vertex1] + travel_time, instance.tw_close[vertex2])
            return (t0 - t1).total_seconds()

    # w_ij represents the minimal waiting time (due to time window restrictions) at one of the locations if a vehicle
    # serves both locations immediately after each other
    w_ij = max(0, min(waiting_time_seconds(vertex1, vertex2), waiting_time_seconds(vertex2, vertex1)))
    gamma = 2
    return gamma * ut.travel_time(instance.distance([vertex1], [vertex2])).total_seconds() + w_ij


def los_schulte_request_similarity(delivery0, pickup0, delivery1, pickup1, vertex_similarity_matrix):
    """
    Following Los, J., Schulte, F., Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020).
    Decentralized combinatorial auctions for dynamic and large-scale collaborative vehicle routing.

    :param delivery0:
    :param pickup0:
    :param delivery1:
    :param pickup1:
    :param vertex_similarity_matrix:
    :return:
    """
    return min(
        vertex_similarity_matrix[pickup0][delivery1],
        vertex_similarity_matrix[delivery0][pickup1],
        0.5 * (vertex_similarity_matrix[pickup0][pickup1] + vertex_similarity_matrix[delivery0][delivery1])
    )


def ropke_pisinger_request_similarity(instance: it.MDPDPTWInstance,
                                      solution: slt.CAHDSolution,
                                      request_0: int,
                                      request_1: int,
                                      distance_weight: float = 1,
                                      time_weight: float = 1,
                                      capacity_weight: float = 1,
                                      num_vehicles_weight: float = 1):
    """
    Following Ropke, Stefan, & Pisinger, David. (2006). An Adaptive Large Neighborhood Search Heuristic for the
    Pickup and Delivery Problem with Time Windows. Transportation Science, 40(4), 455–472.
    As far as I can see, it is not useful to precompute a request similarity matrix since the similarity is based
    on the arrival times and these arrival times obviously change with each execute neighborhood move

    :param request_0:
    :param request_1:
    :param distance_weight:
    :param time_weight:
    :param capacity_weight:
    :param num_vehicles_weight:
    :return:
    """
    pickup_0, delivery_0 = instance.pickup_delivery_pair(request_0)
    pickup_1, delivery_1 = instance.pickup_delivery_pair(request_1)
    distance_max = max(max(x) for x in instance._distance_matrix)
    distance_min = min(min(x) for x in instance._distance_matrix)
    normalized_distance_matrix = [ut.linear_interpolation(x, 0, 1, distance_min, distance_max)
                                  for x in instance._distance_matrix]
    distance = normalized_distance_matrix[pickup_0][pickup_1] + normalized_distance_matrix[delivery_0][delivery_1]

    # TODO slow search algorithm. I intentionally avoided to have a request_to_tour assignment array in the solution
    #  because keeping it updated meant ugly dependencies on the CAHDSolution class for many functions
    tour_0, tour_1 = [solution.tour_of_request(r) for r in (request_0, request_1)]

    pickup_pos_0 = tour_0.vertex_pos[pickup_0]
    delivery_pos_0 = tour_0.vertex_pos[delivery_0]
    pickup_pos_1 = tour_1.vertex_pos[pickup_1]
    delivery_pos_1 = tour_1.vertex_pos[delivery_1]

    arrival_time_delta = (abs(tour_0.arrival_time_sequence[pickup_pos_0] - tour_1.arrival_time_sequence[pickup_pos_1]) +
                          abs(tour_0.arrival_time_sequence[delivery_pos_0] - tour_1.arrival_time_sequence[
                              delivery_pos_1])
                          ).total_seconds()
    # normalize time [0, 1]
    start_time_delta = (ut.EXECUTION_START_TIME - dt.datetime.min).total_seconds()
    end_time_delta = (ut.END_TIME - dt.datetime.min).total_seconds()
    # ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    arrival_time_delta = (arrival_time_delta - start_time_delta) / (end_time_delta - start_time_delta * (1 - 0)) + 0

    normalized_load = ut.linear_interpolation(instance.vertex_load, 0, 1)
    capacity = abs(normalized_load[pickup_0] - normalized_load[pickup_1])

    # num_vehicles not relevant in my application since I assume that all vertices can be served by all vehicles.
    # num_vehicles = ...

    return distance_weight * distance + time_weight * arrival_time_delta + capacity_weight * capacity  # + num_vehicles_weight * num_vehicles


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
# BUNDLING EVALUATION CLASSES
# combine the above functions in a certain way
# ======================================================================================================================

class BundlingValuation(ABC):
    """
    Class to compute the valuation of a bundling based on some valuation measure(s)
    """
    def __init__(self):
        self.name = self.__class__.__name__

    @final
    def evaluate_bundling_labels(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                                 bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """turns bundling labels into bundling and evaluates that bundling"""
        bundling = bundling_labels_to_bundling(bundling_labels, auction_request_pool)
        return self.evaluate_bundling(instance, solution, bundling)

    @abstractmethod
    def evaluate_bundling(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """evaluate a bundling"""
        pass

    def preprocessing(self, instance: it.MDPDPTWInstance, auction_request_pool: Sequence[int]):
        pass


class GHProxyBundlingValuation(BundlingValuation):
    """
    The quality of a bundling is defined as: \n
    (min(isolations) * min(densities)) / (max(total_travel_distances) * num_bundles) \n
    Each of isolations, densities and total_travel_distances is a list of values per bundle in the bundling. The
    total_travel_distance for a bundle is estimated by a very rough proxy function (bundle_total_travel_distance_proxy)

    """

    def evaluate_bundling(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
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
            approx_travel_dist = bundle_total_travel_distance_proxy(instance, bundle)

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

    def evaluate_bundling(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        bundle_valuations = []
        for bundle in bundling:
            valuation = bundle_total_travel_distance(instance, solution, bundle)
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class LosSchulteBundlingValuation(BundlingValuation):
    """
    uses the request similarity measure by Los et al. (2020) to compute a clustering evaluation measure (cohesion)
    """

    def evaluate_bundling(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:

        bundle_valuations = []
        for bundle in bundling:
            bundle_valuation = self.evaluate_bundle(instance, bundle)
            bundle_valuations.append(bundle_valuation)

        # compute mean valuation of the bundles in the bundling; lower values are better
        # TODO try max, min or other measures instead of mean?
        bundling_valuation = sum(bundle_valuations) / len(bundle_valuations)

        # return inverse, since low values are better and the caller maximizes
        return 1 / bundling_valuation

    def evaluate_bundle(self, instance: it.MDPDPTWInstance, bundle: Sequence[int]):
        # multi-request bundles
        if len(bundle) > 1:
            # lower request_similarity values are better
            request_relatedness_list = []

            for i, request0 in enumerate(bundle[:-1]):
                p0, d0 = instance.pickup_delivery_pair(request0)
                # adjust vertex indices to account for depots
                p0 -= instance.num_carriers
                d0 -= instance.num_carriers

                for j, request1 in enumerate(bundle[i + 1:]):
                    p1, d1 = instance.pickup_delivery_pair(request1)
                    # adjust vertex indices to account for depots
                    p1 -= instance.num_carriers
                    d1 -= instance.num_carriers

                    # compute request_similarity between requests 0 and 1 acc. to the paper's formula (1)
                    request_similarity = los_schulte_request_similarity(d0, p0, d1, p1, self.vertex_relatedness_matrix)

                    request_relatedness_list.append(request_similarity)

            # collect mean request_similarity of the requests in the bundle; lower values are better
            # TODO try max, min or other measures instead of mean?
            # TODO: cluster weights acc. to cluster size?
            bundle_valuation = sum(request_relatedness_list) / len(bundle)

        # single-request bundles
        else:
            p0, d0 = instance.pickup_delivery_pair(bundle[0])
            # adjust vertex indices to account for depots
            p0 -= instance.num_carriers
            d0 -= instance.num_carriers
            # todo: cluster weights acc. to cluster size?
            bundle_valuation = self.vertex_relatedness_matrix[p0][d0]

        return bundle_valuation

    def preprocessing(self, instance: it.MDPDPTWInstance, auction_request_pool: Sequence[int]):
        """
        pre-compute the pairwise relatedness matrix for all vertices
        """

        n = instance.num_requests

        self.vertex_relatedness_matrix = [[0.0] * n * 2 for _ in range(n * 2)]

        for i, request1 in enumerate(instance.requests):
            pickup1, delivery1 = instance.pickup_delivery_pair(request1)
            for j, request2 in enumerate(instance.requests):
                pickup2, delivery2 = instance.pickup_delivery_pair(request2)

                # [1] pickup1 <> delivery1
                self.vertex_relatedness_matrix[i][i + n] = los_schulte_vertex_similarity(instance, pickup1, delivery1)

                # [2] pickup1 <> pickup2
                self.vertex_relatedness_matrix[i][j] = los_schulte_vertex_similarity(instance, pickup1, pickup2)

                # [3] pickup1 <> delivery2
                self.vertex_relatedness_matrix[i][j + n] = los_schulte_vertex_similarity(instance, pickup1, delivery2)

                # [4] delivery1 <> pickup2
                self.vertex_relatedness_matrix[i + n][j] = los_schulte_vertex_similarity(instance, delivery1, pickup2)

                # [5] delivery1 <> delivery2
                self.vertex_relatedness_matrix[i + n][j + n] = los_schulte_vertex_similarity(instance, delivery1,
                                                                                             delivery2)

                # [6] pickup2 <> delivery2
                self.vertex_relatedness_matrix[j][j + n] = los_schulte_vertex_similarity(instance, pickup2, delivery2)


class RandomBundlingValuation(BundlingValuation):
    def evaluate_bundling(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        return random.random()


class BundlingValuation2(ABC):
    """
    attempt at implementing a new & improved BundlingValuation interface/architecture.

    """

    def __init__(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        self._vertex_distance_matrix = instance._distance_matrix
        self._request_distance_matrix = self._compute_request_distance_matrix(instance, solution)

        # todo: a _bundles_distance_matrix would make it so that the instance of this class is tied to a specific
        #  bundling. Thus, for each bundling, the same request_distance_matrix will have to be recalculated
        # self._bundles_distance_matrix = self._compute_bundles_distance_matrix()

    @abstractmethod
    def evaluate_bundling(self, bundling: Sequence[Sequence[int]]):
        pass

    @abstractmethod
    def evaluate_bundle(self, bundle: Sequence[int]):
        pass

    def _compute_request_distance_matrix(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        request_distance_matrix = []
        for i, request0 in enumerate(instance.requests[:-1]):
            request_distance_array = []
            for j, request1 in instance.requests[i:]:
                distance = self.request_distance(instance, solution, request0, request1)
                request_distance_array.append(distance)
            request_distance_matrix.append(request_distance_array)
        return request_distance_matrix

    @abstractmethod
    def request_distance(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, request0: int, request1: int):
        """
        Computes a pairwise distance matrix of requests based on the given distance function.

        :param instance:
        :param solution:
        :param request0:
        :param request1:
        :return:
        """
        pass