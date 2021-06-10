import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm
import multiprocessing

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_improvement as imp
from src.cr_ahd.utility_module.utils import ConstraintViolationError

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def execute(self,
                instance: it.PDPInstance,
                solution: slt.CAHDSolution,
                bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        returns a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle:

        """
        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=True):
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
                      solution: slt.CAHDSolution,
                      bundle: Sequence[int],
                      carrier: int):
        pass


class CheapestInsertionDistanceIncrease(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: List[int], carrier: int):
        # travel distance without the bundle
        without_bundle = solution.carriers[carrier].sum_travel_distance()

        # calculate the travel distance with the bundle's request included. Must happen from scratch to ensure the same
        # result as in the acceptance phase

        # create a temporary copy to compute the correct bids
        tmp_carrier = instance.num_carriers
        tmp_carrier_ = deepcopy(solution.carriers[carrier])

        # reset the temporary carrier's solution and start from scratch instead
        tmp_carrier_.tours.clear()
        tmp_carrier_.assigned_requests.extend(bundle)
        tmp_carrier_.assigned_requests.sort()  # must be sorted due to dynamism
        tmp_carrier_.unrouted_requests.extend(tmp_carrier_.assigned_requests)
        solution.carriers.append(tmp_carrier_)
        solution.carrier_depots.append(solution.carrier_depots[carrier])

        try:
            construction = cns.CheapestPDPInsertion()
            while tmp_carrier_.unrouted_requests:
                request = tmp_carrier_.unrouted_requests[0]
                insertion = construction._carrier_cheapest_insertion(instance,
                                                                     solution,
                                                                     tmp_carrier,
                                                                     [request]  # one at a time
                                                                     )
                request, tour, pickup_pos, delivery_pos = insertion

                # when for a given request no tour can be found, create a new tour. This may raise a
                # ConstraintViolationError if the carrier (a) would exceed fleet size constraints or (b) a pendulum
                # tour would exceed time window constraints
                if tour is None:
                    construction._create_new_tour_with_request(instance, solution, tmp_carrier, request)

                else:
                    construction._execute_insertion(instance, solution, tmp_carrier, request, tour, pickup_pos,
                                                    delivery_pos)

                # local search
                imp.PDPMove().improve_carrier_solution(instance, solution, tmp_carrier, False)
                imp.PDPTwoOpt().improve_carrier_solution(instance, solution, tmp_carrier, False)
                imp.PDPRelocate().improve_carrier_solution(instance, solution, tmp_carrier, False)

            with_bundle = tmp_carrier_.sum_travel_distance()

        except ConstraintViolationError:
            with_bundle = float('inf')

        finally:
            solution.carriers.pop()  # del the temporary carrier copy
            solution.carrier_depots.pop()  # del the tmp_carrier's depot

        return with_bundle - without_bundle


class Profit(BiddingBehavior):
    def _generate_bid(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: List[int], carrier: int):
        ins_cost = CheapestInsertionDistanceIncrease()._generate_bid(instance, solution, bundle, carrier)
        ins_revenue = 0
        for r in bundle:
            pickup, delivery = instance.pickup_delivery_pair(r)
            ins_revenue += instance.revenue[pickup] + instance.revenue[delivery]
        return ins_revenue - ins_cost
