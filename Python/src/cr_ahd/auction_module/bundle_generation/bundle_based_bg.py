from abc import ABC, abstractmethod
from typing import Sequence
from core_module import instance as it, solution as slt
from utility_module import utils as ut
import bundle_generation as bg
import auction_module.partition_valuation as pv


class LimitedNumBundles(bg.BundleGeneration):
    def __init__(self, num_auction_bundles: int, bundle_valuation: bv.BundleValuation, **kwargs):
        """
        
        :param num_auction_bundles: number of bundles in the auction bundle pool 
        :param bundle_valuation: value function for assessing the quality of a bundle
        :param kwargs: further parameters as a dict
        """
        super().__init__()
        self.num_auction_bundles = num_auction_bundles
        self.bundle_valuation = bundle_valuation
        self.parameters = kwargs

    @abstractmethod
    def _generate_auction_bundles(self,
                                  instance: it.MDVRPTWInstance,
                                  solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        pass

    def preprocessing(self, instance: it.MDVRPTWInstance, auction_request_pool: Sequence[int]):
        self.bundling_valuation.preprocessing(instance, auction_request_pool)
        pass


class AllBundles(bg.BundleGeneration):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set.
    """

    def _generate_auction_bundles(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        return tuple(ut.power_set(range(len(auction_request_pool)), False))


class BestOfAllBundles(bg.BundleGeneration):
    """
    Generates the power set of all the submitted requests and selects those that have the best valuation according
    to the bundle valuation function
    """

    def _generate_auction_bundles(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        all_bundles = tuple(ut.power_set(range(len(auction_request_pool)), False))
        best_bundles = sorted(all_bundles, key=self.bundle_valuation)[:self.num_bundles]
