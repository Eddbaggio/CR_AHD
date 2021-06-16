import datetime as dt
from typing import Sequence, List, Tuple

import numpy as np
from scipy.spatial.distance import squareform, pdist

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_initialization as ini, tour_construction as cns
from src.cr_ahd.utility_module import utils as ut


def bundle_direct_travel_dist(instance: it.PDPInstance, bundle: Sequence):
    """
    direct travel distances between pickup-delivery pairs of requests in the bundle
    """
    direct_travel_dist = []
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        direct_travel_dist.append(instance.distance([pickup], [delivery]))
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
        centers.append(ut.midpoint_(instance.x_coords[pickup], instance.y_coords[pickup],
                                    instance.x_coords[delivery], instance.y_coords[delivery]))

    # there is one instance (run=1+dist=200+rad=150+n=10) for which pickup and delivery of a request are at the same
    # location, in that case, the pd_direct_travel_dist is zero, thus the following if clause is needed
    if sum(pd_direct_travel_dist) == 0:
        weights = None
    else:
        weights = pd_direct_travel_dist

    centroid = ut.Coordinates(*np.average(centers, axis=0, weights=weights))
    return centroid


def bundle_extended_distance_matrix(instance: it.PDPInstance, additional_vertex_coords: Sequence[ut.Coordinates]):
    x = instance.x_coords.copy()
    y = instance.y_coords.copy()
    for centroid in additional_vertex_coords:
        x.append(centroid.x)
        y.append(centroid.y)
    return squareform(pdist(np.array(list(zip(x, y)))))


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
        # the actual index for the distance matrix
        idx = instance.num_depots + 2 * instance.num_requests + bundle_idx
        travel_dist_to_centroid.append((extended_distance_matrix[pickup][idx], extended_distance_matrix[delivery][idx]))
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
                                       bundle: Sequence[int],
                                       ):
    """
    estimate the total travel distance required to visit all requests in the bundle. Ignores all constraints (time
    windows, vehicle capacity, max tour length, ...)
    """
    routing_sequence = []
    for request in bundle:
        routing_sequence.extend(instance.pickup_delivery_pair(request))
    return instance.distance(routing_sequence[:-1], routing_sequence[1:])


def bundle_total_travel_distance(instance: it.PDPInstance,
                                 solution: slt.CAHDSolution,
                                 centroid: ut.Coordinates,
                                 bundle_idx: int,
                                 bundle: Sequence[int],
                                 extended_distance_matrix
                                 ):
    """
    total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic
    ["...using an algorithm proposed by Renaud et al. (2000).]\n
    BEFORE: used bundle centroid as depot\n
    NOW: uses bundle member as depot, because the former version may lead to infeasibility.

    :param extended_distance_matrix: a replica of the instance's distance matrix, extended by all the distances to the centroids that exist in the candidate solution. these additional distances are positioned at the very end of the matrix; the centroids will have the corresponding indices starting at num_carriers + 2* num_requests
    """
    num_bundles = len(extended_distance_matrix) - (instance.num_depots + 2 * instance.num_requests)
    tour_dict = {'routing_sequence': [],
                 'travel_distance_sequence': [],
                 'travel_duration_sequence': [],
                 'load_sequence': [],
                 'revenue_sequence': [],
                 'arrival_schedule': [],
                 'service_schedule': []}
    instance_dict = {
        'num_depots': instance.num_depots,
        'num_requests': instance.num_requests,
        'distance_matrix': extended_distance_matrix,  # using the extended version
        'vertex_load': instance.load + [0] * num_bundles,
        'revenue': instance.revenue + [0] * num_bundles,
        'service_duration': instance.service_duration + [dt.timedelta(0)] * num_bundles,
        'vehicles_max_travel_distance': instance.vehicles_max_travel_distance,
        'vehicles_max_load': instance.vehicles_max_load,
    }
    solution_dict = {
        'tw_open': solution.tw_open + [ut.START_TIME] * num_bundles,
        'tw_close': solution.tw_close + [ut.END_TIME] * num_bundles,
    }

    # treat the pickup of the first-to-open delivery vertex as the depot

    # TODO not sure why i have to initialize in reverse order here, but I do the same in the Tour class and doing it
    #  with [0, 1] causes errors whn updating
    tr.insert_and_update(indices=[1], vertices=[artificial_depot], **tour_dict, **instance_dict, **solution_dict)
    tr.insert_and_update(indices=[0], vertices=[artificial_depot], **tour_dict, **instance_dict, **solution_dict)

    # furthest distance seed request
    best_evaluation = -float('inf')
    initialization = ini.FurthestDistance()
    tmp_bundle = bundle[:]
    for request_idx, request in enumerate(tmp_bundle):
        evaluation = initialization._request_evaluation(*instance.pickup_delivery_pair(request),
                                                        x_coords=instance.x_coords, y_coords=instance.y_coords,
                                                        x_depot=centroid.x, y_depot=centroid.y)
        if evaluation > best_evaluation:
            seed = request
            seed_idx = request_idx
            best_evaluation = evaluation
    tmp_bundle = np.delete(tmp_bundle, seed_idx)
    tr.insert_and_update(indices=(1, 2), vertices=instance.pickup_delivery_pair(seed), **tour_dict, **instance_dict,
                         **solution_dict)

    # construction
    while tmp_bundle.any():
        best_request = None
        best_request_idx = None
        best_delta = float('inf')
        best_pickup_pos = None
        best_delivery_pos = None
        for request_idx, request in enumerate(tmp_bundle):
            delta, pickup_pos, delivery_pos = cns.tour_cheapest_insertion(*instance.pickup_delivery_pair(request),
                                                                          **tour_dict, **instance_dict, **solution_dict)
            if delta < best_delta:
                best_request = request
                best_request_idx = request_idx
                best_delta = delta
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

            # if not feasible to insert this request into the bundle-tour (initializing a second route: cannot ensure
            # feasibility)
            elif delta == float('inf'):
                return float('inf')
        tr.insert_and_update(indices=(best_pickup_pos, best_delivery_pos),
                             vertices=instance.pickup_delivery_pair(best_request), **tour_dict, **instance_dict,
                             **solution_dict)
        tmp_bundle = np.delete(tmp_bundle, best_request_idx)

    return sum(tour_dict['travel_distance_sequence'])


def GHProxyValuation(instance: it.PDPInstance,
                     solution: slt.CAHDSolution,
                     candidate_solution: Sequence[int],
                     auction_pool: Sequence[int]
                     ):
    """

    :param auction_pool:
    :param instance:
    :param solution:
    :param candidate_solution: encoded as a sequence of bundle indices
    :return:
    """
    centroids = []
    request_direct_travel_distances = []
    num_bundles = max(candidate_solution) + 1
    candidate_solution = np.array(candidate_solution)
    auction_pool_array = np.array(auction_pool)
    for bundle_idx in range(num_bundles):
        bundle = auction_pool_array[candidate_solution == bundle_idx]
        pd_direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
        request_direct_travel_distances.append(pd_direct_travel_dist)
        centroid = bundle_centroid(instance, bundle, pd_direct_travel_dist)
        centroids.append(centroid)

    extended_distance_matrix = bundle_extended_distance_matrix(instance, centroids)

    radii = []
    densities = []
    total_travel_distances = []
    for bundle_idx in range(num_bundles):
        # recreate the bundle
        # todo make the functions below work with the GH decoding of bundles directly?
        bundle = auction_pool_array[candidate_solution == bundle_idx]

        # compute the radius
        travel_dist_to_centroid = bundle_vertex_to_centroid_travel_dist(instance, bundle_idx, bundle,
                                                                        extended_distance_matrix)
        radii.append(bundle_radius(bundle, travel_dist_to_centroid))

        # compute bundle density
        densities.append(bundle_density(request_direct_travel_distances[bundle_idx], travel_dist_to_centroid))

        # estimating the tour length of the bundle
        approx_travel_dist = bundle_total_travel_distance_proxy(instance, bundle)

        # if there is no feasible tour for this bundle, return a valuation of negative infinity for the whole bundling
        if approx_travel_dist == float('inf'):
            return -float('inf')

        total_travel_distances.append(approx_travel_dist)

    isolations = []
    for bundle_idx in range(num_bundles):
        isolations.append(bundle_isolation(centroids[bundle_idx],
                                           centroids[:bundle_idx] + centroids[bundle_idx + 1:],
                                           radii[bundle_idx],
                                           radii[:bundle_idx] + radii[bundle_idx + 1:]))

    return (min(isolations) * min(densities)) / (max(total_travel_distances) * max(candidate_solution)+1)
