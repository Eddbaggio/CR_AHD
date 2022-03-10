from typing import Sequence

from core_module import instance as it, tour as tr
from routing_module import tour_construction as cns, metaheuristics as mh, neighborhoods as nh
from utility_module import utils as ut
from utility_module.utils import argmin


def bundle_centroid(instance: it.CAHDInstance, bundle: Sequence):
    """
    centroid of the requests
    """
    x = [instance.vertex_x_coords[instance.vertex_from_request(r)] for r in bundle]
    y = [instance.vertex_y_coords[instance.vertex_from_request(r)] for r in bundle]
    return ut.Coordinates(sum(x) / len(x), sum(y) / len(y))


def bundle_sum_squared_errors(instance: it.CAHDInstance, bundle: Sequence[int], centroid: ut.Coordinates):
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


def bundle_cohesion_centroid_based(instance: it.CAHDInstance, bundle: Sequence[int], centroid: ut.Coordinates):
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


def bundle_cohesion_graph_based(instance: it.CAHDInstance, bundle: Sequence[int]):
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


def bundle_vertex_to_centroid_travel_dist(instance: it.CAHDInstance,
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


def bundle_total_travel_distance_proxy(instance: it.CAHDInstance, bundle: Sequence[int]):
    """
    VERY rough estimate for the total travel distance required to visit all requests in the bundle. Ignores all
    constraints (time windows, vehicle capacity, max tour length, ...)
    """
    routing_sequence = []
    for request in bundle:
        routing_sequence.append(instance.vertex_from_request(request))
    return instance.travel_distance(routing_sequence[:-1], routing_sequence[1:])


def bundle_total_travel_distance(instance: it.CAHDInstance, bundle: Sequence[int]):
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
    tmp_tour_ = tr.Tour('tmp', depot_vertex)

    # insert all remaining requests of the bundle
    tour_construction = cns.VRPTWMinTravelDistanceInsertion()  # TODO this should be a parameter!

    for request in sorted(bundle, key=lambda x: instance.request_disclosure_time[x]):
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

    # TODO improvement should be a parameter!
    # tour_improvement = mh.VRPTWVariableNeighborhoodDescent([nh.PDPMove(), nh.PDPTwoOpt()])
    # tour_improvement.execute_on_tour(instance, tmp_tour_)

    return tmp_tour_.sum_travel_distance


def bundle_total_travel_duration(instance: it.CAHDInstance, bundle: Sequence[int]):
    """
    total *travel* duration needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic

    uses bundle member vertex as depot. Does not check feasibility of the tour

    """

    # treat the first-to-open delivery vertex as the depot
    depot_request = argmin([instance.tw_open[instance.vertex_from_request(r)] for r in bundle])
    depot_request = bundle[depot_request]

    # initialize temporary tour with the earliest request
    depot_vertex = instance.vertex_from_request(depot_request)
    tmp_tour_ = tr.Tour('tmp', depot_vertex)

    # insert all remaining requests of the bundle
    tour_construction = cns.VRPTWMinTravelDurationInsertion()  # TODO this should be a parameter!

    for request in sorted(bundle, key=lambda x: instance.request_disclosure_time[x]):
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

    # TODO improvement should be a parameter!
    # tour_improvement = mh.VRPTWVariableNeighborhoodDescent([nh.VRPTWMoveDur(), nh.VRPTWTwoOptDur()], 2.0)
    # tour_improvement.execute_on_tour(instance, tmp_tour_)

    return tmp_tour_.sum_travel_duration
