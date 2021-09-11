import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic):
        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.name = self.__class__.__name__

    def execute_bidding(self,
                        instance: it.MDPDPTWInstance,
                        solution: slt.CAHDSolution,
                        bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        :return a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle

        """

        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=True):
            carrier_bundle_bids = []
            for carrier in solution.carriers:
                logger.debug(f'Carrier {carrier.id_} generating bids for bundle {b}')

                value_without_bundle = carrier.objective()
                value_with_bundle = self._value_with_bundle(instance, solution, bundles[b], carrier.id_)
                bid = value_with_bundle - value_without_bundle

                # REMOVEME
                # if all([x in range(carrier.id_*instance.num_requests_per_carrier, carrier.id_*instance.num_requests_per_carrier+instance.num_requests_per_carrier) for x in bundles[b]]):
                #     print('\n')

                carrier_bundle_bids.append(bid)

            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @staticmethod
    def _add_bundle_to_carrier(bundle: Sequence[int], carrier: slt.AHDSolution):
        """
        add the bundle to the carriers assigned, accepted and unrouted requests
        sorting is required to ensure that dynamic insertion order is the same as in the acceptance phase (in
        particular, this is important for when a carrier computes the bid on his own original bundle)
        """

        carrier.assigned_requests.extend(bundle)
        carrier.assigned_requests.sort()

        carrier.accepted_requests.extend(bundle)
        carrier.accepted_requests.sort()

        carrier.unrouted_requests.extend(bundle)
        carrier.unrouted_requests.sort()

    @abstractmethod
    def _value_with_bundle(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier_id: int):
        pass


class DynamicInsertionAndImprove(BiddingBehavior):
    """
    The profit for the carrier WITH the bundle added is calculated by inserting all requests (the ones that already
    belong to the carrier AND the bundle's requests) sequentially in their order of vertex index, i.e. in the order
    of request arrival to mimic the dynamic request arrival process within the acceptance phase. Afterwards, an
    improvement is executed using the defined metaheuristic.
    """

    def _value_with_bundle(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier_id: int):

        solution_copy = deepcopy(solution)
        carrier_copy = solution_copy.carriers[carrier_id]
        self._add_bundle_to_carrier(bundle, carrier_copy)

        # reset the temporary carrier's solution and start from scratch instead
        carrier_copy.clear_routes()

        # sequentially insert requests (original + bundle) just as if it was the acceptance phase
        try:
            while carrier_copy.unrouted_requests:
                request = carrier_copy.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution_copy, carrier_copy.id_, request)

            solution_copy_improved = self.tour_improvement.execute(instance, solution_copy, [carrier_id])
            carrier_copy_improved = solution_copy_improved.carriers[carrier_copy.id_]

            with_bundle = carrier_copy_improved.objective()

        except ut.ConstraintViolationError:
            with_bundle = -float('inf')

        return with_bundle
