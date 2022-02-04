import datetime as dt
import random
from abc import ABC, abstractmethod
from typing import Sequence, List, final

import numpy as np
from scipy.spatial.distance import squareform, pdist

from core_module import instance as it, solution as slt, tour as tr
from routing_module import tour_construction as cns, metaheuristics as mh, \
    neighborhoods as nh
from utility_module import utils as ut
# ======================================================================================================================
# STAND-ALONE BUNDLE EVALUATION MEASURES
# ======================================================================================================================
from utility_module.utils import argmin


def bundle_centroid(instance: it.MDVRPTWInstance, bundle: Sequence):
    """
    centroid of the requests
    """
    x = [instance.vertex_x_coords[instance.vertex_from_request(r)] for r in bundle]
    y = [instance.vertex_y_coords[instance.vertex_from_request(r)] for r in bundle]
    return ut.Coordinates(sum(x) / len(x), sum(y) / len(y))


def bundle_sum_squared_errors(instance: it.MDVRPTWInstance, bundle: Sequence[int], centroid: ut.Coordinates):
    """based on cluster analysis evaluation"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    vertex_sse_list = []
    for request in bundle:
        # todo: use the request's midpoint rather than pickup and delivery individually
        for vertex in instance.pickup_delivery_pair(request):
            # todo maybe better use the extended_distance_matrix that includes distances to centroids rather than
            #  computing it all individually
            vertex_sse_list.append(ut.euclidean_distance(*instance.coords(vertex), *centroid))
    bundle_sse = sum(vertex_sse_list) / len(vertex_sse_list)
    return bundle_sse


def bundle_cohesion_centroid_based(instance: it.MDVRPTWInstance, bundle: Sequence[int], centroid: ut.Coordinates):
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


def bundle_cohesion_graph_based(instance: it.MDVRPTWInstance, bundle: Sequence[int]):
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


def bundling_extended_distance_matrix(instance: it.MDVRPTWInstance, additional_vertex_coords: Sequence[ut.Coordinates]):
    """
    uses euclidean distance for distances to/from additional_vertex_coords

    :param instance:
    :param additional_vertex_coords:
    :return:
    """
    x = instance.vertex_x_coords.copy()
    y = instance.vertex_y_coords.copy()
    for centroid in additional_vertex_coords:
        x.append(centroid.x)
        y.append(centroid.y)
    # NOTE: must use ceil function to ensure triangle inequality
    return np.ceil(squareform(pdist(np.array(list(zip(x, y)))))).astype('int')


def bundle_vertex_to_centroid_travel_dist(instance: it.MDVRPTWInstance,
                                          bundle_idx: int,
                                          bundle: Sequence,
                                          extended_distance_matrix):
    """
    euclidean distances of all delivery points in the bundle to the bundle’s centroid.

    """
    travel_dist_to_centroid = []
    for request in bundle:
        delivery = instance.vertex_from_request(request)
        # the centroid index for the distance matrix
        centroid_idx = instance.num_carriers + instance.num_requests + bundle_idx
        travel_dist_to_centroid.append(extended_distance_matrix[delivery][centroid_idx])
    return travel_dist_to_centroid


def bundle_radius(bundle: Sequence, vertex_to_centroid_travel_dist: Sequence[float]):
    """average distance of all delivery points in the bundle to the bundle’s centroid."""
    return sum(vertex_to_centroid_travel_dist) / len(vertex_to_centroid_travel_dist)


def bundle_density(direct_travel_dist, vertex_to_centroid_travel_dist: Sequence[float]):
    """average direct travel distance of all requests in the bundle, divided by the maximum of the distances of all
    requests to the bundle’s centroid. The distance of a request to the bundle’s centroid is determined by the sum of
    the distances of its pickup point and its delivery point to the centroid.
     """
    raise NotImplementedError()
    avg_direct_travel_dist = sum(direct_travel_dist) / len(
        direct_travel_dist)  # fixme there is no such thing as direct_travel_dist in VRP!
    max_dist_to_centroid = max(vertex_to_centroid_travel_dist)
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


def bundle_total_travel_distance_proxy(instance: it.MDVRPTWInstance, bundle: Sequence[int]):
    """
    VERY rough estimate for the total travel distance required to visit all requests in the bundle. Ignores all
    constraints (time windows, vehicle capacity, max tour length, ...)
    """
    routing_sequence = []
    for request in bundle:
        routing_sequence.append(instance.vertex_from_request(request))
    return instance.travel_distance(routing_sequence[:-1], routing_sequence[1:])


def bundle_total_travel_distance(instance: it.MDVRPTWInstance, bundle: Sequence[int]):
    """
    total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic
    ["...using an algorithm proposed by Renaud et al. (2000).]\n
    uses bundle member vertex as depot. Does not check feasibility of the tour

    """

    # treat the first-to-open delivery vertex as the depot
    depot_request = argmin([instance.tw_open[instance.vertex_from_request(r)] for r in bundle])
    depot_request = bundle[depot_request]

    # initialize temporary tour with the earliest request
    depot_vertex = instance.vertex_from_request(depot_request)
    tmp_tour_ = tr.VRPTWTour('tmp', depot_vertex)

    # insert all remaining requests of the bundle
    tour_construction = cns.VRPTWMinTravelDistanceInsertion()  # TODO this should be a parameter!
    tour_improvement = mh.VRPTWVariableNeighborhoodDescent(
        [nh.PDPMove(), nh.PDPTwoOpt()])  # TODO this should be a parameter!
    for request in bundle:
        if request == depot_request:
            continue
        delivery = instance.vertex_from_request(request)

        # set check_feasibility to False since only the tour length is of interest here
        delta, delivery_pos = tour_construction.best_insertion_for_request_in_tour(instance, tmp_tour_,
                                                                                   request, check_feasibility=False)

        # if no feasible insertion exists, return inf for the travel distance
        if delivery_pos is None:
            return float('inf')

        # insert, ignores feasibility!
        tmp_tour_.insert_and_update(instance, [delivery_pos], [delivery])
    tour_improvement.execute_on_tour(instance, tmp_tour_)
    return tmp_tour_.sum_travel_distance


def bundle_total_travel_duration(instance: it.MDVRPTWInstance, bundle: Sequence[int]):
    """
    total travel duration needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic

    uses bundle member vertex as depot. Does not check feasibility of the tour

    """

    # treat the first-to-open delivery vertex as the depot
    depot_request = argmin([instance.tw_open[instance.vertex_from_request(r)] for r in bundle])
    depot_request = bundle[depot_request]

    # initialize temporary tour with the earliest request
    depot_vertex = instance.vertex_from_request(depot_request)
    tmp_tour_ = tr.VRPTWTour('tmp', depot_vertex)

    # insert all remaining requests of the bundle
    tour_construction = cns.VRPTWMinTravelDurationInsertion()  # TODO this should be a parameter!
    tour_improvement = mh.VRPTWVariableNeighborhoodDescent(
        [nh.VRPTWMoveDur(), nh.VRPTWTwoOptDur()], 2.0)  # TODO this should be a parameter!
    for request in bundle:
        if request == depot_request:
            continue
        delivery = instance.vertex_from_request(request)

        # set check_feasibility to False since only the tour length is of interest here
        delta, delivery_pos = tour_construction.best_insertion_for_request_in_tour(instance, tmp_tour_,
                                                                                   request, check_feasibility=False)

        # if no feasible insertion exists, return inf for the travel duration
        if delivery_pos is None:
            return float('inf')

        # insert, ignores feasibility!
        tmp_tour_.insert_and_update(instance, [delivery_pos], [delivery])
    tour_improvement.execute_on_tour(instance, tmp_tour_)
    return tmp_tour_.sum_travel_duration


def waiting_time_seconds(instance: it.MDVRPTWInstance, vertex1: int, vertex2: int):
    travel_time = instance.travel_duration([vertex1], [vertex2])
    if instance.tw_open[vertex1] + travel_time > instance.tw_close[vertex2]:
        return float('inf')
    else:
        t0 = max(instance.tw_open[vertex1] + travel_time, instance.tw_open[vertex2])  # earliest start of service at v2
        t1 = min(instance.tw_close[vertex1] + travel_time, instance.tw_close[vertex2])  # latest start of service at v2
        return (t0 - t1).total_seconds()


def los_schulte_vertex_similarity(instance: it.MDVRPTWInstance, vertex1: int, vertex2: int):
    """
    Following [1] Los, J., Schulte, F., Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020).
    Decentralized combinatorial auctions for dynamic and large-scale collaborative vehicle routing.

    similarity of two vertices (not requests!) based on a weighted sum of travel time and waiting time
    """

    # w_ij represents the minimal waiting time (due to time window restrictions) at one of the locations, if a vehicle
    # serves both locations immediately after each other
    w_ij = max(0,
               min(waiting_time_seconds(instance, vertex1, vertex2), waiting_time_seconds(instance, vertex2, vertex1)))
    avg_travel_dur = instance.travel_duration([vertex1, vertex2], [vertex2, vertex1]) / 2
    gamma = 2
    return gamma * avg_travel_dur.total_seconds() + w_ij


def los_schulte_request_similarity(delivery0, delivery1, vertex_similarity_matrix):
    """
    Following Los, J., Schulte, F., Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020).
    Decentralized combinatorial auctions for dynamic and large-scale collaborative vehicle routing.

    :param delivery0:
    :param delivery1:
    :param vertex_similarity_matrix:
    :return:
    """
    raise NotImplementedError('Not yet implemented for VRP')
    return min(
        vertex_similarity_matrix[pickup0][delivery1],
        vertex_similarity_matrix[delivery0][pickup1],
        0.5 * (vertex_similarity_matrix[pickup0][pickup1] + vertex_similarity_matrix[delivery0][delivery1])
    )


def ropke_pisinger_request_similarity(instance: it.MDVRPTWInstance,
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
    raise NotImplementedError('Not yet implemented for VRP')
    pickup_0, delivery_0 = instance.pickup_delivery_pair(request_0)
    pickup_1, delivery_1 = instance.pickup_delivery_pair(request_1)
    distance_max = max(max(x) for x in instance._travel_distance_matrix)
    distance_min = min(min(x) for x in instance._travel_distance_matrix)
    normalized_distance_matrix = [ut.linear_interpolation(x, 0, 1, distance_min, distance_max)
                                  for x in instance._travel_distance_matrix]
    distance = normalized_distance_matrix[pickup_0][pickup_1] + normalized_distance_matrix[delivery_0][delivery_1]

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
    def evaluate_bundling_labels(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                                 bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """turns bundling labels into bundling and evaluates that bundling"""
        bundling = ut.indices_to_nested_lists(bundling_labels, auction_request_pool)
        return self.evaluate_bundling(instance, solution, bundling)

    @abstractmethod
    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """evaluate a bundling"""
        pass

    def preprocessing(self, instance: it.MDVRPTWInstance, auction_request_pool: Sequence[int]):
        pass


class GHProxyBundlingValuation(BundlingValuation):
    """
    The quality of a bundling is defined as: \n
    (min(isolations) * min(densities)) / (max(total_travel_distances) * num_bundles) \n
    Each of isolations, densities and total_travel_distances is a list of values per bundle in the bundling. The
    total_travel_distance for a bundle is estimated by a very rough proxy function (bundle_total_travel_distance_proxy)

    """

    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        """

        :return: the bundle's valuation/fitness
        """
        raise NotImplementedError()
        centroids = []
        request_direct_travel_distances = []  # fixme does not exist for VRP
        num_bundles = len(bundling)

        for bundle in bundling:
            # pd_direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
            # request_direct_travel_distances.append(pd_direct_travel_dist)
            centroid = bundle_centroid(instance, bundle)
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
            density = bundle_sum_squared_errors()  # FIXME cannot compute density the way it was done before
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

    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        bundle_valuations = []
        for bundle in bundling:
            valuation = bundle_total_travel_distance(instance, bundle)
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class MinDurationBundlingValuation(BundlingValuation):
    """
    The value of a BUNDLING is determined by the minimum over the travel distances per BUNDLE.
    The travel distance of a bundle is determined by building a route that traverses all the bundle's requests using
    the dynamic insertion procedure.
    """

    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        raise NotImplementedError  # TODO does not work since bundles of size 1 have travel duration = 0
        bundle_valuations = []
        for bundle in bundling:
            valuation = bundle_total_travel_duration(instance, bundle).total_seconds()
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class LosSchulteBundlingValuation(BundlingValuation):
    """
    uses the request similarity measure by Los et al. (2020) to compute a clustering evaluation measure (cohesion)
    VRP: adjusted the formulas and algorithms to be somewhat applicable to a delivery-only setting
    """

    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:

        bundle_valuations = []
        for bundle in bundling:
            bundle_valuation = self.evaluate_bundle(instance, bundle)
            bundle_valuations.append(bundle_valuation)

        # compute mean valuation of the bundles in the bundling; lower values are better
        # TODO try max, min or other measures instead of mean?
        bundling_valuation = sum(bundle_valuations) / len(bundle_valuations)

        # return negative, since low values are better and the caller maximizes
        return 1000 / bundling_valuation

    def evaluate_bundle(self, instance: it.MDVRPTWInstance, bundle: Sequence[int]):
        # multi-request bundles
        if len(bundle) > 1:
            # lower request_similarity values are better
            request_relatedness_list = []

            for i, request0 in enumerate(bundle[:-1]):
                d0 = instance.vertex_from_request(request0)
                # adjust vertex index to account for depots
                d0 -= instance.num_carriers

                for j, request1 in enumerate(bundle[i + 1:]):
                    d1 = instance.vertex_from_request(request1)
                    # adjust vertex index to account for depots
                    d1 -= instance.num_carriers

                    # compute request_similarity between requests 0 and 1 acc. to the paper's formula (1)
                    assert self.vertex_relatedness_matrix[d0][d1] == self.vertex_relatedness_matrix[d1][
                        d0]  # REMOVEME they acutally should not always be equal due to asymmetric travel times, remove once this has been confirmed
                    request_similarity = self.vertex_relatedness_matrix[d0][
                        d1]  # + self.vertex_relatedness_matrix[d1][d0] ) *0.5

                    request_relatedness_list.append(request_similarity)

            # collect mean request_similarity of the requests in the bundle; lower values are better
            # TODO try max, min or other measures instead of mean?
            # TODO: cluster weights acc. to cluster size?
            bundle_valuation = sum(request_relatedness_list) / len(bundle)

        # single-request bundles
        else:
            bundle_valuation = 0  # FIXME maybe not 100% appropriate: will distort the mean calculated by the caller fct

        return bundle_valuation

    def preprocessing(self, instance: it.MDVRPTWInstance, auction_request_pool: Sequence[int]):
        """
        pre-compute the pairwise relatedness matrix for all vertices
        """
        self.vertex_relatedness_matrix = [[0.0] * instance.num_requests for _ in range(instance.num_requests)]
        for i, request1 in enumerate(instance.requests):
            delivery1 = instance.vertex_from_request(request1)
            for j, request2 in enumerate(instance.requests):
                if i == j:
                    continue
                delivery2 = instance.vertex_from_request(request2)
                self.vertex_relatedness_matrix[i][j] = los_schulte_vertex_similarity(instance, delivery1, delivery2)


class RandomBundlingValuation(BundlingValuation):
    def evaluate_bundling(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                          bundling: List[List[int]]) -> float:
        return random.random()
