from core_module import instance as it, solution as slt

from abc import ABC, abstractmethod
from typing import Sequence


class BundleGeneration(ABC):
    """
    Generates auction bundles based on partitioning the auction request pool. This guarantees a feasible solution
    for the Winner Determination Problem which cannot (easily) be guaranteed if bundles are generated without
    considering that the WDP requires a partitioning of the auction request pool (see bundle_based_bg.py)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    def execute(self,
                instance: it.MDVRPTWInstance,
                solution: slt.CAHDSolution,
                auction_request_pool: Sequence[int],
                original_partition_labels: Sequence[int]):
        self.preprocessing(instance, auction_request_pool)
        auction_bundle_pool = self._generate_auction_bundles(instance, solution, auction_request_pool,
                                                             original_partition_labels)
        return auction_bundle_pool

    def preprocessing(self, instance, auction_request_pool):
        pass

    @abstractmethod
    def _generate_auction_bundles(self, instance, solution, auction_request_pool, original_partition_labels):
        pass
