import itertools
import logging
from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Container, List

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut
from src.cr_ahd.auction_module import bundle_valuation as bv

logger = logging.getLogger(__name__)


class BundleSetGenerationBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        bundle_set = self._generate_bundle_set(instance, solution, auction_pool)
        solution.bundle_generation = self.__class__.__name__
        return bundle_set

    @abstractmethod
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        pass


class AllBundles(BundleSetGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set.
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        return tuple(ut.power_set(auction_pool, False))


class RandomMaxKPartition(BundleSetGenerationBehavior):
    """
    creates a random partition of the submitted bundles with AT MOST as many subsets as there are carriers
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        bundles = ut.random_max_k_partition(auction_pool, max_k=instance.num_carriers)
        return bundles


class KMeansBundles(BundleSetGenerationBehavior):
    """
    creates a k-means partitions of the submitted requests. generates exactly as many clusters as there are carriers.

    :return a List of lists, where each sublist contains the cluster members of a cluster
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Iterable):
        request_midpoints = [ut.midpoint(instance, *instance.pickup_delivery_pair(sr)) for sr in auction_pool]
        # k_means = KMeans(n_clusters=instance.num_carriers, random_state=0).fit(request_midpoints)
        k_means = KMeans(n_clusters=instance.num_carriers).fit(request_midpoints)
        bundles = [list(np.take(auction_pool, np.nonzero(k_means.labels_ == b)[0])) for b in
                   range(k_means.n_clusters)]
        return bundles


class GeneticAlgorithm(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        population = self._initialize_population(auction_pool)
        fitness = bv.GHProxyValuation(instance, solution, population)

        pass

    @staticmethod
    def _normalize_individual(individual: List[int]):
        mapping_idx = 0
        mapping: List[int] = [-1] * len(individual)
        mapping[individual[0]] = mapping_idx
        individual[0] = mapping_idx

        for i in range(1, len(individual)):
            if mapping[individual[i]] is None:
                mapping_idx += 1
                mapping[individual[i]] = mapping_idx
                individual[i] = mapping_idx
            else:
                individual[i] = mapping[individual[i]]
        pass

    def _initialize_population(self, auction_pool: Sequence[int]):
        population_size = 100
        n = len(auction_pool)
        population = [ut.random_max_k_partition_idx(auction_pool, n) for _ in range(population_size)]
        for individual in population:
            self._normalize_individual(individual)
        return population

    def _roulette_wheel(self, fitness:Sequence[float]):
        pass

    def _crossover_uniform(self, parent1, parent2):
        pass




class ProxyTest(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        best_candidate_solution = None
        best_proxy_valuation = 0

        # candidate == auction pool
        candidate = [auction_pool]
        valuation = bv.GHProxyValuation(instance, solution, [auction_pool])
        if valuation > best_proxy_valuation:
            best_candidate_solution = candidate
            best_proxy_valuation = valuation

        # generate all partitions of size [2, 3, 4, ... instance.num_carriers] i.e. all possible candidate solutions
        for k in range(2, instance.num_carriers + 1):
            candidate_generator = algorithm_u(auction_pool, k)
            for candidate in tqdm(candidate_generator, total=stirling_second(len(auction_pool), instance.num_carriers)):
                valuation = bv.GHProxyValuation(instance, solution, candidate)

                if valuation > best_proxy_valuation:
                    best_candidate_solution = candidate
                    best_proxy_valuation = valuation

        return best_candidate_solution


class ProxyTest2(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        candidate_generator = RandomMaxKPartition()
        best_candidate_solution = None
        best_proxy_valuation = 0

        runs = 0
        pbar = tqdm(total=runs + 1)
        while best_proxy_valuation <= 0:
            candidate = candidate_generator.execute(instance, solution, auction_pool)
            valuation = bv.GHProxyValuation(instance, solution, candidate)

            if valuation > best_proxy_valuation:
                best_candidate_solution = candidate
                best_proxy_valuation = valuation

            runs += 1
            pbar.update(1)
        pbar.close()
        return best_candidate_solution


def stirling_second(n, k):
    """
    Stirling numbers of the second kind: number of ways to partition a set of n objects into m non-empty subsets
    https://extr3metech.wordpress.com/2013/01/21/stirling-number-generation-using-python-code-examples/
    Stirling Algorithm
    Cod3d by EXTR3ME
    https://extr3metech.wordpress.com
    """
    n1 = n
    k1 = k
    if n <= 0:
        return 1

    elif k <= 0:
        return 0

    elif n == 0 and k == 0:
        return -1

    elif n != 0 and n == k:
        return 1

    elif n < k:
        return 0

    else:
        temp1 = stirling_second(n1 - 1, k1)
        temp1 = k1 * temp1
        return (k1 * (stirling_second(n1 - 1, k1))) + stirling_second(n1 - 1, k1 - 1)


def algorithm_u(ns, m):
    """
    https://codereview.stackexchange.com/a/1944

    Generates all set partitions with a given number of blocks. The total amount of k-partitions is given by the
     Stirling number of the second kind

    :param ns: sequence of integers to build the subsets from
    :param m: integer, smaller than ns, number of subsets
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
    pass
