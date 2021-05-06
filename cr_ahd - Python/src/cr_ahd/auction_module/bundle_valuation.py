from copy import deepcopy

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut
from src.cr_ahd.routing_module import tour_construction as cns, tour_improvement as imp, tour_initialization as ini
from typing import Sequence
import numpy as np


def bundle_centroid(instance: it.PDPInstance, bundle: Sequence, direct_travel_dist: Sequence):
    """centroid of the request’s centers, where the center of request is the midpoint between pickup and delivery
     location. centers are weighted with the length of their request, which is the direct travel distance between
      pickup and delivery of request"""
    centers = (ut.midpoint(*instance.pickup_delivery_pair(request)) for request in bundle)
    centroid = ut.Coordinates(*np.average(centers, axis=0, weights=direct_travel_dist))
    return centroid


def bundle_direct_travel_dist(instance: it.PDPInstance, bundle: Sequence):
    return [instance.distance([pickup], [delivery])
            for request in bundle for pickup, delivery in instance.pickup_delivery_pair(request)]


def bundle_travel_dist_to_centroid(instance: it.PDPInstance, bundle: Sequence, centroid):
    return (ut._euclidean_distance(ut.Coordinates(instance.x_coords[pickup], instance.y_coords[pickup]), centroid) +
            ut._euclidean_distance(ut.Coordinates(instance.x_coords[deliv], instance.y_coords[deliv]), centroid)
            for request in bundle for pickup, deliv in instance.pickup_delivery_pair(request))


def bundle_radius(bundle: Sequence, travel_dist_to_centroid):
    """average distance of all points in the bundle (pickup and delivery) to the bundle’s centroid."""
    return sum(travel_dist_to_centroid) / len(bundle)


def bundle_density(direct_travel_dist, travel_dist_to_centroid):
    """average direct travel distance of all requests in the bundle, divided by the maximum of the distances of all
    requests to the bundle’s centroid. """
    avg_direct_travel_dist = sum(direct_travel_dist) / len(direct_travel_dist)
    max_dist_to_centroid = max(travel_dist_to_centroid)
    return avg_direct_travel_dist / max_dist_to_centroid


def bundle_separation(centroid_a, centroid_b, radius_a, radius_b):
    """approximates the separation of two bundles a and b."""
    centroid_dist = ut._euclidean_distance(centroid_a, centroid_b)
    return centroid_dist / max(radius_a, radius_b)


def bundle_isolation(bundle_centroid, other_bundles_centroid, bundle_radius, other_bundles_radius):
    """minimum separation of bundle from other bundles"""
    return min(bundle_separation(bundle_centroid, other_centroid, bundle_radius, other_radius)
               for other_centroid, other_radius in zip(other_bundles_centroid, other_bundles_radius))


def bundle_tour_length(instance: it.PDPInstance, solution: slt.GlobalSolution, centroid: ut.Coordinates, bundle: Sequence):
    """total travel distance needed to visit all requests in a bundle. Since finding the optimal solutions for
    all bundles is too time consuming, the tour length is approximated using the cheapest insertion heuristic
    ["...using an algorithm proposed by Renaud et al. (2000).] """

    # furthest distance seed request
    for request in bundle:
        pickup, delivery = instance.pickup_delivery_pair(request)
        midpoint = ut.midpoint_(instance.coords(pickup), instance.coords(delivery))
        dist = ut._euclidean_distance(centroid, midpoint)



    carrier_ = solution.carriers[carrier]
    tmp_carrier_ = deepcopy(carrier_)
    tmp_carrier_.unrouted_requests = bundle[:]
    tmp_carrier = instance.num_carriers
    solution.carriers.append(tmp_carrier_)

    try:
        construction = cns.CheapestPDPInsertion()
        while tmp_carrier_.unrouted_requests:
            request, tour, pickup_pos, delivery_pos = \
                construction._carrier_cheapest_insertion(instance, solution, tmp_carrier,
                                                         tmp_carrier_.unrouted_requests)

            # when for a given request no tour can be found, create a new tour and start over. This may raise
            # a ConstraintViolationError if the carrier cannot initialize another new tour
            if tour is None:
                construction._create_new_tour_with_request(instance, solution, tmp_carrier, request)

            else:
                construction._execute_insertion(instance, solution, tmp_carrier, request, tour, pickup_pos,
                                                delivery_pos)
        imp.PDPMoveBest().improve_carrier_solution(instance, solution, tmp_carrier)
        tour_length = tmp_carrier_.sum_travel_distance()
    except ut.ConstraintViolationError:
        tour_length = float('inf')
    finally:
        solution.carriers.pop()
    return tour_length


def GHProxy(instance: it.PDPInstance, solution: slt.GlobalSolution, candidate_solution: Sequence):
    candidate_valuation = 0

    centroids = []
    radii = []
    densities = []
    tour_lengths = []
    for bundle in candidate_solution:
        direct_travel_dist = bundle_direct_travel_dist(instance, bundle)
        centroid = bundle_centroid(instance, bundle, direct_travel_dist)
        centroids.append(centroid)

        travel_dist_to_centroid = bundle_travel_dist_to_centroid(instance, bundle, centroid)
        radii.append(bundle_radius(bundle, travel_dist_to_centroid))

        densities.append(bundle_density(direct_travel_dist, travel_dist_to_centroid))

        tour_length = bundle_tour_length(instance, solution, centroid, bundle)


    isolations = []
    for i in range(len(candidate_solution)):
        isolations.append(bundle_isolation(centroids[i], centroids[:i] + centroids[i + 1:], radii[i],
                                           radii[:i] + radii[i + 1:]))

    for

        pass

    def algorithm_u(ns, m):
        """
        https://codereview.stackexchange.com/a/1944

        Generates all set partitions with a given number of blocks

        :param ns: sequence of integers
        :param m: integer, smaller than ns
        :return:
        """
        assert m > 1

        def visit(n, a):
            ps = [[] for i in range(m)]
            for j in range(n):
                ps[a[j + 1]].append(ns[j])
            return ps

        def f(mu, nu, sigma, n, a):
            if mu == 2:
                yield visit(n, a)
            else:
                for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v
            if nu == mu + 1:
                a[mu] = mu - 1
                yield visit(n, a)
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    yield visit(n, a)
            elif nu > mu + 1:
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = mu - 1
                else:
                    a[mu] = mu - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v

        def b(mu, nu, sigma, n, a):
            if nu == mu + 1:
                while a[nu] < mu - 1:
                    yield visit(n, a)
                    a[nu] = a[nu] + 1
                yield visit(n, a)
                a[mu] = 0
            elif nu > mu + 1:
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] < mu - 1:
                    a[nu] = a[nu] + 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = 0
                else:
                    a[mu] = 0
            if mu == 2:
                yield visit(n, a)
            else:
                for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v

        n = len(ns)
        a = [0] * (n + 1)
        for j in range(1, m + 1):
            a[n - m + j] = j - 1
        return f(m, n, 0, n, a)

    if __name__ == '__main__':
        for v in algorithm_u([1, 3, 5, 2, 4, 6], 1):
            print(v)
