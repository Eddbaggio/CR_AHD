import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.utility_module.utils import ConstraintViolationError

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def execute(self,
                instance: it.PDPInstance,
                solution: slt.GlobalSolution,
                bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        returns a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle:

        """
        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding'):
            carrier_bundle_bids = []
            for carrier in range(instance.num_carriers):
                logger.debug(f'Carrier {carrier} generating bids for bundle {b}')
                carrier_bundle_bids.append(self._generate_bid(instance, solution, bundles[b], carrier))
            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @abstractmethod
    def _generate_bid(self,
                      instance: it.PDPInstance,
                      solution: slt.GlobalSolution,
                      bundle: Sequence[int],
                      carrier: int):
        pass


class CheapestInsertionDistanceIncrease(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundle: List[int], carrier: int):
        without_bundle = solution.carriers[carrier].sum_travel_distance()

        # TODO: can I avoid the copy here? Only if I make cns.CheapestInsertion()._carrier_cheapest_insertion return
        #  the delta for the cheapest insertion of ALL (!) the requests in the bundle, which is quite impossible because
        #  the cheapest insertion of one request in the bundle depends on the previous insertion
        #  Alternatively, add the bundle to the original carrier, do the routing, and remove it again?!
        tmp_carrier = instance.num_carriers
        tmp_carrier_ = deepcopy(solution.carriers[carrier])
        tmp_carrier_.unrouted_requests.extend(bundle)
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

            with_bundle = tmp_carrier_.sum_travel_distance()
        except ConstraintViolationError:
            with_bundle = float('inf')
        finally:
            solution.carriers.pop()  # del the temporary carrier copy
        return with_bundle - without_bundle


class Profit(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundle: List[int], carrier: int):
        ins_cost = CheapestInsertionDistanceIncrease()._generate_bid(instance, solution, bundle, carrier)
        ins_revenue = 0
        for r in bundle:
            pickup, delivery = instance.pickup_delivery_pair(r)
            ins_revenue += instance.revenue[pickup] + instance.revenue[delivery]
        return ins_revenue - ins_cost
