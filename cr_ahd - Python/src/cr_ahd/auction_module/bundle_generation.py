import itertools
import logging
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans

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


class RandomPartition(BundleSetGenerationBehavior):
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
        k_means = KMeans(n_clusters=instance.num_carriers, random_state=0).fit(request_midpoints)
        bundles = [list(np.take(auction_pool, np.nonzero(k_means.labels_ == b)[0])) for b in
                   range(k_means.n_clusters)]
        return bundles




class GHProxySetPacking(BundleSetGenerationBehavior):
    """
    using bundle valuation measure of Gansterer & Hartl. Does not use a Genetic Algorithm!
    Instead evaluates all bundles with the measures ... to obtain candidate solutions... ?
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        bundles = AllBundles()._generate_bundle_set(instance, solution, auction_pool)
        bundle_valuation = []
        for bundle in bundles:
            bv.gansterer_hartl_proxy()


class GHProxySetPacking2(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence):
        candidate_solutions =
        candidate_valuations = []
        for candidate in candidate_solutions:
            candidate_valuations.append(bv.GHProxy(instance, solution, candidate))
