import datetime as dt
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm
from gurobipy import GRB

import utility_module.errors
from core_module import instance as it, solution as slt
from routing_module import tour_construction as cns, metaheuristics as mh
from utility_module import utils as ut

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def __init__(self,
                 tour_construction: cns.VRPTWInsertionConstruction,
                 tour_improvement: mh.VRPTWMetaHeuristic):
        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.name = self.__class__.__name__

    def execute_bidding(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
                        bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        :return a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle

        """

        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=not ut.debugger_is_active()):
            bundle = bundles[b]
            carrier_bundle_bids = []
            for carrier in solution.carriers:
                logger.debug(f'Carrier {carrier.id_} generating bids for bundle {b}={bundle}')

                value_without_bundle = carrier.objective()
                value_with_bundle = self._value_with_bundle(instance, solution, bundle, carrier.id_)
                bid = value_with_bundle - value_without_bundle

                carrier_bundle_bids.append(bid)

            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @staticmethod
    def _add_bundle_to_carrier(instance: it.CAHDInstance, carrier: slt.AHDSolution, bundle: Sequence[int]):
        """
        add the bundle to the carriers assigned, accepted and unrouted requests.
        correct sorting is required to ensure that dynamic insertion order is the same as in the acceptance phase. in
        particular, this is important for when a carrier computes the bid on his own original bundle.

        Example for the sorting issue:
        before request selection: [0, 1, 2, 3, 4, 5]
        after request selection: [0, 1, 4, 5]
        without sorting, dynamic insertion for computing the bid would happen in the order [0, 1, 4, 5, 2, 3] which
        may make it impossible to reach the original ask price
        """
        carrier.assigned_requests.extend(bundle)
        carrier.assigned_requests.sort(key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))

        carrier.accepted_requests.extend(bundle)
        carrier.accepted_requests.sort(key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))

        carrier.unrouted_requests.extend(bundle)
        carrier.unrouted_requests.sort(key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))

        # THE BELOW APPROACH CAUSES INCONSISTENT BIDDING RESULTS THAT ARE BELOW THE PRE-REQUEST-SELECTION SOLUTION!
        # carrier_post_rs.assigned_requests = [r for r in carrier_pre_rs.assigned_requests if
        #                                      r in carrier_post_rs.assigned_requests + bundle]
        # carrier_post_rs.assigned_requests += [r for r in bundle if r not in carrier_pre_rs.assigned_requests]
        #
        # carrier_post_rs.accepted_requests = [r for r in carrier_pre_rs.accepted_requests if
        #                                      r in carrier_post_rs.accepted_requests + bundle]
        # carrier_post_rs.accepted_requests += [r for r in bundle if r not in carrier_pre_rs.accepted_requests]
        #
        # carrier_post_rs.unrouted_requests = [r for r in carrier_pre_rs.unrouted_requests if
        #                                      r in carrier_post_rs.unrouted_requests + bundle]
        # carrier_post_rs.unrouted_requests += [r for r in bundle if r not in carrier_pre_rs.unrouted_requests]

        pass

    @abstractmethod
    def _value_with_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier_id: int):
        pass


class ClearAndReinsertAll(BiddingBehavior):
    """
    The profit for the carrier WITH the bundle added is calculated by inserting all requests (the ones that already
    belong to the carrier AND the bundle's requests) sequentially in their order of vertex index, i.e. in the order
    of request arrival to mimic the dynamic request arrival process within the acceptance phase. Afterwards, an
    improvement is executed using the defined metaheuristic.
    Note that this is only compatible if the same dynamic re-optimization is also applied after the reallocation, too.
    """

    def _value_with_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier_id: int):

        solution_copy = deepcopy(solution)
        carrier_copy = solution_copy.carriers[carrier_id]
        self._add_bundle_to_carrier(instance, carrier_copy, bundle)

        # reset the temporary carrier's solution and start from scratch instead
        solution_copy.clear_carrier_routes([carrier_copy.id_])

        # sequentially insert requests (original + bundle) just as if it was the acceptance phase
        try:
            for request in sorted(carrier_copy.unrouted_requests, key=lambda x: instance.request_disclosure_time[x]):
                self.tour_construction.insert_single_request(instance, solution_copy, carrier_copy.id_, request)

            solution_copy_improved = self.tour_improvement.execute(instance, solution_copy, [carrier_id])
            carrier_copy_improved = solution_copy_improved.carriers[carrier_copy.id_]

            with_bundle = carrier_copy_improved.objective()

        except utility_module.errors.ConstraintViolationError:
            # with_bundle = -float('inf')
            if isinstance(solution.objective(), float):
                with_bundle = -GRB.INFINITY
            elif isinstance(solution.objective(), dt.timedelta):
                with_bundle = dt.timedelta.max
        return with_bundle
