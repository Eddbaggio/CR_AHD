from abc import ABC, abstractmethod
from typing import Dict, List, Iterable

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.vehicle import Vehicle
from src.cr_ahd.core_module.vertex import midpoint, Vertex
from src.cr_ahd.solving_module.tour_construction import I1Insertion
from src.cr_ahd.solving_module.tour_initialization import EarliestDueDate
from src.cr_ahd.utility_module.utils import Coordinates, flatten_dict_of_lists, euclidean_distance, \
    InsertionError, random_max_k_partition


class BundleSetGenerationBehavior(ABC):
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def execute(self, submitted_requests: Dict[Carrier, List[Vertex]]):
        """

        :param submitted_requests:
        :return: Dict[int, List[Vertex]]
        """
        bundle_set = self._generate_bundle_set(submitted_requests)
        # solution.bundle_generation = self.__class__.__name__
        return bundle_set

    @abstractmethod
    def _generate_bundle_set(self, submitted_requests: Dict[Carrier, List[Vertex]]) -> Dict[int, Iterable[Vertex]]:
        pass


'''
class AllBundles(BundleGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set!
    """

    # TODO how does this ensure that ALL the submitted requests will be re-assigned? There's no way to make sure that
    #  the union of the bundles that carriers win are actually exhausting the auction pool

    def _generate_bundles(self, submitted_requests: dict):
        pool = flatten_dict_of_lists(submitted_requests)
        return powerset(pool, False)'''


class RandomPartition(BundleSetGenerationBehavior):
    """
    creates a random partition of the submitted bundles with AT MOST number_of_carriers subsets
    """
    def _generate_bundle_set(self, submitted_requests: Dict[Carrier, List[Vertex]]) -> Dict[int, Iterable[Vertex]]:
        pool = flatten_dict_of_lists(submitted_requests)
        num_carriers = len(submitted_requests)
        bundle_set = random_max_k_partition(pool, max_k=num_carriers)
        return bundle_set


class KMeansBundles(BundleSetGenerationBehavior):
    """
    creates a k-means partitions of the submitted requests with len(submitted_requests) locally clustered subsets
    """
    def _generate_bundle_set(self, submitted_requests: Dict[Carrier, List[Vertex]]) -> Dict[int, Iterable[Vertex]]:
        flattened = flatten_dict_of_lists(submitted_requests)
        X = [x.coords for x in flattened]
        k = len(submitted_requests)
        k_means = KMeans(n_clusters=k, random_state=0).fit(X)
        bundle_set = {i: tuple(flattened[j] for j in range(len(flattened)) if i == k_means.labels_[j]) for i in range(k)}
        return bundle_set


class GanstererProxyBundles(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, submitted_requests: Dict[Carrier, List[Vertex]]) -> Dict[int, Iterable[Vertex]]:

        pass

    def bundle_centroid(self, bundle, depot) -> Coordinates:
        """centroid of the request’s centers, where the center of request is the midpoint between pickup and delivery
         location. centers are weighted with the length of their request, which is the direct travel distance between
          pickup and delivery of request"""
        centers = [midpoint(request, depot) for request in bundle]
        weights = [self.distance_matrix[request.id_][depot.id_] for request in bundle]
        centroid = Coordinates(*np.average(tuple(centers), axis=0, weights=weights))
        return centroid

    def bundle_radius(self, bundle,
                      depot):  # todo change signature to accept centroid as an input to avoid redundant function calls
        """is the average distance of all points in the bundle (pickup and delivery) to the bundle’s centroid."""
        centroid = self.bundle_centroid(bundle, depot)
        dist_sum = 0
        for request in bundle:
            dist_sum += euclidean_distance(request.coords, centroid)
        radius = dist_sum / len(bundle)
        return radius

    def bundle_density(self, bundle, depot):
        """average direct travel distance of all requests in the bundle, divided by the maximum of the distances of all
        requests to the bundle’s centroid. """
        avg_travel_dist = sum([self.distance_matrix[request.id_][depot.id_] for request in bundle]) / len(bundle)
        # TODO dist to centroid is calculated multiple times. Maybe it would be wise to have a lookup table to avoid
        #  redundant distance computations
        centroid = self.bundle_centroid(bundle, depot)
        max_dist_to_centroid = max(euclidean_distance(request.coords, centroid) for request in bundle)
        return avg_travel_dist / max_dist_to_centroid

    def bundle_separation(self, bundle_a, bundle_b, depot):
        """approximates the separation from other bundles."""
        centroid_a = self.bundle_centroid(bundle_a, depot)
        centroid_b = self.bundle_centroid(bundle_b, depot)
        centroid_distance = euclidean_distance(centroid_a, centroid_b)
        radius_a = self.bundle_radius(bundle_a, depot)
        radius_b = self.bundle_radius(bundle_b, depot)
        separation = centroid_distance / max(radius_a, radius_b)
        return separation

    def bundle_isolation(self, bundle, other_bundles, depot):
        """minimum separation of bundle from other bundles"""
        isolation = min([self.bundle_separation(bundle, other, depot) for other in other_bundles])
        return isolation

    def bundle_tour_length(self, bundle, depot):
        """total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
        all bundles is too time consuming, the tour length is approximated using the I1 insertion heuristic
        ["...using an algorithm proposed by Renaud et al. (2000).] """
        vehicle = Vehicle('tmp', np.infty)
        carrier = Carrier('tmp', depot, [vehicle], bundle, self.distance_matrix)
        c = True
        while c:
            try:
                I1Insertion().solve_carrier(carrier)
                c = False
            except InsertionError:
                EarliestDueDate().initialize_carrier(carrier)

        tour_length = carrier.cost()
        carrier.retract_requests_and_update_routes(carrier.requests)
        return tour_length
