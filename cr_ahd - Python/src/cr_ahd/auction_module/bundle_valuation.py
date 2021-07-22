import datetime as dt
import random
from abc import ABC, abstractmethod
from math import ceil
from typing import Sequence, Tuple
from typing import Iterable
import numpy as np
from scipy.spatial.distance import squareform, pdist

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_initialization as ini, tour_construction as cns, metaheuristics as mh, \
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

def bundle_cohesion_graph_based(instance:it.PDPInstance, bundle:Sequence[int]):
    """values closer to 1 are better"""
    raise NotImplementedError('Bundle evaluation inspired by external cluster evaluation measures not yet complete')
    # since the bundle contains *request* IDs, they need to be unpacked to vertex IDs
    bundle_vertices = []
    for request in bundle:
        bundle_vertices.extend(instance.pickup_delivery_pair(request))

    pair_distances = []
    for vertex_i in bundle_vertices[:-1]:
        for vertex_j in bundle_vertices[1:]:
            pair_distances.append(ut.euclidean_distance(*instance.coords(vertex_i), *instance.coords(vertex_j)))

    cohesion = 1 - (sum(pair_distances))/(len(bundle)*(len(bundle)-1))
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


class BundleValuation(ABC):
    @abstractmethod
    def evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                        bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        pass


class GHProxyBundleValuation(BundleValuation):
    @pr.timing
    def evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                        bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """

        :param auction_request_pool:
        :param instance:
        :param solution:
        :param bundling_labels: encoded as a sequence of bundle indices
        :return: the bundle's valuation/fitness
        """
        centroids = []
        request_direct_travel_distances = []
        num_bundles = max(bundling_labels) + 1
        bundling_labels = np.array(bundling_labels)
        auction_request_pool_array = np.array(auction_request_pool)
        for bundle_idx in range(num_bundles):
            bundle = auction_request_pool_array[bundling_labels == bundle_idx]
            pd_direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
            request_direct_travel_distances.append(pd_direct_travel_dist)
            centroid = bundle_centroid(instance, bundle, pd_direct_travel_dist)
            centroids.append(centroid)

        extended_distance_matrix = bundling_extended_distance_matrix(instance, centroids)

        radii = []
        densities = []
        total_travel_distances = []
        for bundle_idx in range(num_bundles):
            # todo make the functions called below work with the GH decoding of bundles directly?
            # recreate the bundle from the bundling_labels
            bundle = auction_request_pool_array[bundling_labels == bundle_idx]

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
                return -float('inf')

            total_travel_distances.append(approx_travel_dist)

        isolations = []
        for bundle_idx in range(num_bundles):
            isolation = bundle_isolation(centroids[bundle_idx], centroids[:bundle_idx] + centroids[bundle_idx + 1:],
                                         radii[bundle_idx], radii[:bundle_idx] + radii[bundle_idx + 1:])
            isolations.append(isolation)

        evaluation = (min(isolations) * min(densities)) / (max(total_travel_distances) * max(bundling_labels) + 1)
        return evaluation


class MyProxyBundleValuation(BundleValuation):
    def evaluate_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                        bundling_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """consider time window information to assess the value of a given bundling."""

        auction_pool_array = np.array(auction_request_pool)
        for bundle_idx in range(max(bundling_labels) + 1):
            # recreate the bundle
            bundle = auction_pool_array[bundling_labels == bundle_idx]

            # find a suitable depot: pickup vertex of the most urgent (earliest TW) delivery vertex
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
            for request in bundle:
                if request == depot_request:
                    continue
                tour_construction = cns.MinTravelDistanceInsertion()
                delta, pickup_pos, delivery_pos = tour_construction.best_insertion_for_request_in_tour(instance,
                                                                                                       solution, tour_,
                                                                                                       request)
                pickup, delivery = instance.pickup_delivery_pair(request)

                # insert, ignores feasibility!
                tour_.insert_and_update(instance, solution, [pickup_pos, delivery_pos], [pickup, delivery])
            pass
