import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_initialization as ini, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic):
        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement

    def execute_bidding(self,
                        instance: it.PDPInstance,
                        solution: slt.CAHDSolution,
                        bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        :return a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle

        """

        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=True):
            # assert [b[i] < b[i+1] for i in range(len(b)-1]  # make sure bundles are sorted?
            carrier_bundle_bids = []
            for carrier in range(instance.num_carriers):
                logger.debug(f'Carrier {carrier} generating bids for bundle {b}')

                value_without_bundle = solution.carriers[carrier].sum_profit()
                value_with_bundle = self._value_with_bundle(instance, solution, bundles[b], carrier)
                bid = value_with_bundle - value_without_bundle

                carrier_bundle_bids.append(bid)

            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    def _create_tmp_carrier_copy_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             bundle: Sequence[int], carrier: int):
        # create a temporary copy to compute the correct bids
        tmp_carrier = instance.num_carriers  # ID
        tmp_carrier_ = deepcopy(solution.carriers[carrier])

        # add the bundle to the carriers assigned and accepted requests
        tmp_carrier_.assigned_requests.extend(bundle)
        tmp_carrier_.assigned_requests.sort()

        tmp_carrier_.accepted_requests.extend(bundle)
        tmp_carrier_.accepted_requests.sort()

        tmp_carrier_.unrouted_requests.extend(bundle)
        tmp_carrier_.unrouted_requests.sort()

        solution.carriers.append(tmp_carrier_)
        solution.carrier_depots.append(solution.carrier_depots[carrier])

        return tmp_carrier, tmp_carrier_

    @abstractmethod
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):
        pass


class DynamicInsertion(BiddingBehavior):
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):
        tmp_carrier, tmp_carrier_ = self._create_tmp_carrier_copy_with_bundle(instance, solution, bundle, carrier)

        # insert bundle's requests 'dynamically', i.e. in their order inside the tmp_carrier_.unrouted list
        try:
            while tmp_carrier_.unrouted_requests:
                request = tmp_carrier_.unrouted_requests[0]
                self.tour_construction.insert_single(instance, solution, tmp_carrier, request)
            with_bundle = tmp_carrier_.sum_profit()
        except ut.ConstraintViolationError:
            with_bundle = -float('inf')
        finally:
            solution.carriers.pop()  # del the temporary carrier copy
            solution.carrier_depots.pop()  # del the tmp_carrier's depot

        return with_bundle


class DynamicInsertionAndImprove(BiddingBehavior):
    """
    The profit for the carrier WITH the bundle added is calculated by inserting all requests (the ones that already
    belong to the carrier AND the bundle's requests) sequentially in their order of vertex index, i.e. in the order
    of request arrival to mimic the dynamic request arrival process within the acceptance phase. Afterwards, an
    improvement is executed using the defined metaheuristic.
    """
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):

        # create temporary copy of the carrier
        tmp_carrier, tmp_carrier_ = self._create_tmp_carrier_copy_with_bundle(instance, solution, bundle, carrier)

        # reset the temporary carrier's solution and start from scratch instead
        tmp_carrier_.clear_routes()

        # assign and insert requests of the bundle
        try:
            while tmp_carrier_.unrouted_requests:
                request = tmp_carrier_.unrouted_requests[0]
                self.tour_construction.insert_single(instance, solution, tmp_carrier, request)
            # start_time = time.time()
            tmp_solution = self.tour_improvement.execute(instance, solution, [tmp_carrier])
            # print(time.time() - start_time)
            with_bundle = tmp_solution.carriers[tmp_carrier].sum_profit()  # todo: CHECK WHETHER THE TMP_CARRIER IS CORRECT! TOUR IMPROVEMENT DOES NOT OPERATE IN PLACE ANY LONGER
        except ut.ConstraintViolationError:
            with_bundle = -float('inf')
        finally:
            solution.carriers.pop()  # del the temporary carrier copy
            solution.carrier_depots.pop()  # del the tmp_carrier's depot

        return with_bundle
