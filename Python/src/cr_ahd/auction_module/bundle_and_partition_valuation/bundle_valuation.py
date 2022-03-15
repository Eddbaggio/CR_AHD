from abc import ABC, abstractmethod
from typing import Sequence, List

from core_module import instance as it, solution as slt


class BundleValuation(ABC):
    """
    Class to compute the valuation of a bundle based on some valuation metric(s)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def evaluate_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, bundle: List[int]):
        """returns the value of a bundle"""
        pass

    def preprocessing(self, instance: it.CAHDInstance, auction_request_pool: Sequence[int]):
        pass


class NoBundleValuation(BundleValuation):

    def evaluate_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, bundle: List[int]):
        pass
