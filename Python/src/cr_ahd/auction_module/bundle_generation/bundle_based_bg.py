from abc import abstractmethod
from typing import Sequence

import utility_module.combinatorics as cmb
from core_module import instance as it, solution as slt
import bundle_gen as bg
from auction_module.bundle_and_partition_valuation import bundle_valuation as bv

'''
class LimitedNumBundles(bg.BundleGenerationBehavior):
    """
    Generate a pool of bundles that has a limited, predefined number of bundles in it.
    Bundle selection happens by evaluating different bundles created from the requests in the auction request pool
    and keeping those bundles with the highest valuation according to some metric (e.g. spatial density)
    """
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
                                  instance: it.CAHDInstance,
                                  solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_partition_labels: Sequence[int]):
        pass

    def preprocessing(self, instance: it.CAHDInstance, auction_request_pool: Sequence[int]):
        self.bundle_valuation.preprocessing(instance, auction_request_pool)
        pass

'''


class AllBundles(bg.BundleGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set.
    """

    def _generate_auction_bundles(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_partition_labels: Sequence[int]):
        return tuple(cmb.power_set(range(len(auction_request_pool)), False))


class BestOfAllBundles(bg.BundleGenerationBehavior):
    """
    Generates the power set of all the submitted requests and selects those that have the best valuation according
    to the bundle valuation function
    """

    def _generate_auction_bundles(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_partition_labels: Sequence[int]):
        all_bundles = tuple(cmb.power_set(range(len(auction_request_pool)), False))
        best_bundles = sorted(all_bundles, key=self.bundle_valuation)[:self.num_bundles]
