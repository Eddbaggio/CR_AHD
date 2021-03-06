import datetime as dt
import random
from abc import ABC, abstractmethod
from typing import Sequence, List, final

import numpy as np
from scipy.spatial.distance import squareform, pdist

import auction_module.bundle_and_partition_valuation.bundle_metrics as bm
from core_module import instance as it, solution as slt
from utility_module import utils as ut
# ======================================================================================================================
# STAND-ALONE BUNDLE EVALUATION MEASURES
# ======================================================================================================================


def partition_extended_distance_matrix(instance: it.CAHDInstance, additional_vertex_coords: Sequence[ut.Coordinates]):
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


def waiting_time_seconds(instance: it.CAHDInstance, vertex1: int, vertex2: int):
    travel_time = instance.travel_duration([vertex1], [vertex2])
    if instance.tw_open[vertex1] + travel_time > instance.tw_close[vertex2]:
        return float('inf')
    else:
        t0 = max(instance.tw_open[vertex1] + travel_time, instance.tw_open[vertex2])  # earliest start of service at v2
        t1 = min(instance.tw_close[vertex1] + travel_time, instance.tw_close[vertex2])  # latest start of service at v2
        return (t0 - t1).total_seconds()


def los_schulte_vertex_similarity(instance: it.CAHDInstance, vertex1: int, vertex2: int):
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


def ropke_pisinger_request_similarity(instance: it.CAHDInstance,
                                      solution: slt.CAHDSolution,
                                      request_0: int,
                                      request_1: int,
                                      distance_weight: float = 1,
                                      time_weight: float = 1,
                                      capacity_weight: float = 1,
                                      num_vehicles_weight: float = 1):
    """
    Following Ropke, Stefan, & Pisinger, David. (2006). An Adaptive Large Neighborhood Search Heuristic for the
    Pickup and Delivery Problem with Time Windows. Transportation Science, 40(4), 455???472.
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
# PARTITION EVALUATION CLASSES
# combine the bundle_metrics and the functions above in a certain way
# ======================================================================================================================

class PartitionValuation(ABC):
    """
    Class to compute the valuation of a partition based on some valuation measure(s)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @final
    def evaluate_partition_labels(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                                  partition_labels: Sequence[int], auction_request_pool: Sequence[int]) -> float:
        """turns partition labels into a partition and evaluates that partition"""
        partition = ut.indices_to_nested_lists(partition_labels, auction_request_pool)
        return self.evaluate_partition(instance, solution, partition)

    @abstractmethod
    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        """evaluate a partition"""
        pass

    def preprocessing(self, instance: it.CAHDInstance, auction_request_pool: Sequence[int]):
        pass


class GHProxyPartitionValuation(PartitionValuation):
    """
    The quality of a partition is defined as: \n
    (min(isolations) * min(densities)) / (max(total_travel_distances) * num_bundles) \n
    Each of isolations, densities and total_travel_distances is a list of values per bundle in the partition. The
    total_travel_distance for a bundle is estimated by a very rough proxy function (bundle_total_travel_distance_proxy)

    """

    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        """

        :return: the bundle's valuation/fitness
        """
        raise NotImplementedError()
        centroids = []
        request_direct_travel_distances = []  # fixme does not exist for VRP
        num_bundles = len(partition)

        for bundle in partition:
            # pd_direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
            # request_direct_travel_distances.append(pd_direct_travel_dist)
            centroid = bm.bundle_centroid(instance, bundle)
            centroids.append(centroid)

        extended_distance_matrix = partition_extended_distance_matrix(instance, centroids)

        radii = []
        densities = []
        total_travel_distances = []

        for bundle_idx, bundle in enumerate(partition):
            # compute the radius
            travel_dist_to_centroid = bm.bundle_vertex_to_centroid_travel_dist(instance, bundle_idx, bundle,
                                                                               extended_distance_matrix)
            radius = bm.bundle_radius(bundle, travel_dist_to_centroid)
            radii.append(radius)

            # compute bundle density
            density = bm.bundle_sum_squared_errors()  # FIXME cannot compute density the way it was done before
            densities.append(density)

            # estimating the tour length of the bundle
            approx_travel_dist = bm.bundle_total_travel_distance_proxy(instance, bundle)

            # if there is no feasible tour for this bundle, return a valuation of negative infinity for the whole
            # partition
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


class MinDistancePartitionValuation(PartitionValuation):
    """
    The value of a partition is determined by the minimum over the travel distances per BUNDLE.
    The travel distance of a bundle is determined by building a route that traverses all the bundle's requests using
    the dynamic insertion procedure.
    """

    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        raise NotImplementedError  # TODO does not work since bundles of size 1 have travel duration = 0
        bundle_valuations = []
        for bundle in partition:
            valuation = bm.bundle_total_travel_distance(instance, bundle)
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class MinTravelDurationPartitionValuation(PartitionValuation):
    """
    The value of a partition is determined by the minimum over the travel distances per BUNDLE.
    The travel distance of a bundle is determined by building a route that traverses all the bundle's requests using
    the dynamic insertion procedure.
    """

    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        raise NotImplementedError  # TODO does not work since bundles of size 1 have travel duration = 0
        bundle_valuations = []
        for bundle in partition:
            valuation = bm.bundle_total_travel_duration(instance, bundle).total_seconds()
            bundle_valuations.append(valuation)
        return 1 / min(bundle_valuations)


class SumTravelDurationPartitionValuation(PartitionValuation):
    """
    The value of a partition is determined by the sum over the *travel* durations per BUNDLE.
    The travel duration of a bundle is determined by building a route that traverses all the bundle's requests using
    the dynamic insertion procedure (without improvement).
    """

    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        bundle_valuations = []
        for bundle in partition:
            valuation = bm.bundle_total_travel_duration(instance, bundle).total_seconds()
            bundle_valuations.append(valuation)
        return sum(bundle_valuations)


class LosSchultePartitionValuation(PartitionValuation):
    """
    uses the request similarity measure by Los et al. (2020) to compute a clustering evaluation measure (cohesion)
    VRP: adjusted the formulas and algorithms to be somewhat applicable to a delivery-only setting
    """

    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:

        bundle_valuations = []
        for bundle in partition:
            bundle_valuation = self.evaluate_bundle(instance, bundle)
            bundle_valuations.append(bundle_valuation)

        # compute mean valuation of the bundles in the partition; lower values are better
        # TODO try max, min or other measures instead of mean?
        partition_valuation = sum(bundle_valuations) / len(bundle_valuations)

        # return negative, since low values are better and the caller maximizes
        return 1000 / partition_valuation

    def evaluate_bundle(self, instance: it.CAHDInstance, bundle: Sequence[int]):
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
                        d0]  # REMOVEME they acutally should not always be equal due to asymmetric travel times. (remove once this has been confirmed)
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

    def preprocessing(self, instance: it.CAHDInstance, auction_request_pool: Sequence[int]):
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


class RandomPartitionValuation(PartitionValuation):
    def evaluate_partition(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                           partition: List[List[int]]) -> float:
        return random.random()
