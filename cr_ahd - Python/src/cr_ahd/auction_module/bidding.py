import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.utility_module.utils import ConstraintViolationError

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundles) -> List[List[float]]:
        """
        returns a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle:

        """
        bundle_bids = []
        for b in range(len(bundles)):
            carrier_bundle_bids = []
            for c in range(instance.num_carriers):
                logger.debug(f'Carrier {c} generating bids for bundle {b}')
                carrier_bundle_bids.append(self._generate_bid(instance, solution, bundles[b], c))
            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @abstractmethod
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundle: List[int], carrier: int):
        pass


class CheapestInsertionDistanceIncrease(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundle: List[int], carrier: int):
        before = solution.carrier_solutions[carrier].sum_travel_distance()
        cs_copy = deepcopy(solution.carrier_solutions[carrier])  # TODO: can I avoid the copy here? Only if I make cns.CheapestInsertion()._carrier_cheapest_insertion return the delta for the cheapest insertion
        cs_copy.unrouted_requests.extend(bundle)
        solution.carrier_solutions.append(cs_copy)
        try:
            cns.CheapestInsertion()._carrier_cheapest_insertion(instance, solution, instance.num_carriers)
            after = cs_copy.sum_travel_distance()
        except ConstraintViolationError:
            after = float('inf')
        finally:
            solution.carrier_solutions.pop()  # del the temporary copy
        return after - before


class Profit(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundle: List[int], carrier: int):
        ins_cost = CheapestInsertionDistanceIncrease()._generate_bid(instance, solution, bundle, carrier)
        ins_revenue = sum([instance.revenue[r] for r in bundle])
        return ins_cost - ins_revenue
