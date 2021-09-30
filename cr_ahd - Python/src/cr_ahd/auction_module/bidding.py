import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

import utility_module.errors
from core_module import instance as it, solution as slt
from routing_module import tour_construction as cns, metaheuristics as mh
from utility_module import utils as ut

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic):
        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.name = self.__class__.__name__

    def execute_bidding(self, instance: it.MDPDPTWInstance, pre_rs_solution: slt.CAHDSolution,
                        post_rs_solution: slt.CAHDSolution, bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        :return a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle

        """

        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=not ut.debugger_is_active()):
            carrier_bundle_bids = []
            for carrier in post_rs_solution.carriers:
                logger.debug(f'Carrier {carrier.id_} generating bids for bundle {b}')

                value_without_bundle = carrier.objective()
                value_with_bundle = self._value_with_bundle(instance, pre_rs_solution, post_rs_solution, bundles[b],
                                                            carrier.id_)
                bid = value_with_bundle - value_without_bundle

                carrier_bundle_bids.append(bid)

            bundle_bids.append(carrier_bundle_bids)
        post_rs_solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @staticmethod
    def _add_bundle_to_carrier(bundle: Sequence[int], carrier_post_rs: slt.AHDSolution,
                               carrier_pre_rs: slt.AHDSolution):
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

        carrier_post_rs.assigned_requests = [r for r in carrier_pre_rs.assigned_requests if
                                             r in carrier_post_rs.assigned_requests + bundle]
        carrier_post_rs.assigned_requests += [r for r in bundle if r not in carrier_pre_rs.assigned_requests]

        carrier_post_rs.accepted_requests = [r for r in carrier_pre_rs.accepted_requests if
                                             r in carrier_post_rs.accepted_requests + bundle]
        carrier_post_rs.accepted_requests += [r for r in bundle if r not in carrier_pre_rs.accepted_requests]

        carrier_post_rs.unrouted_requests = [r for r in carrier_pre_rs.unrouted_requests if
                                             r in carrier_post_rs.unrouted_requests + bundle]
        carrier_post_rs.unrouted_requests += [r for r in bundle if r not in carrier_pre_rs.unrouted_requests]

        pass

    @abstractmethod
    def _value_with_bundle(self, instance: it.MDPDPTWInstance, pre_rs_solution: slt.CAHDSolution,
                           post_rs_solution: slt.CAHDSolution, bundle: Sequence[int], carrier_id: int):
        pass


class InsertBundle(BiddingBehavior):
    """
    The profit for the carrier with the bundle added is calculated by inserting only the bundle's requests into the
    already existing tours. This is only compatible if the same insertion approach is used after the reallocation, too.
    """

    def _value_with_bundle(self, instance: it.MDPDPTWInstance, pre_rs_solution: slt.CAHDSolution,
                           post_rs_solution: slt.CAHDSolution, bundle: Sequence[int], carrier_id: int):
        solution_copy = deepcopy(solution)
        carrier_copy = solution_copy.carriers[carrier_id]
        self._add_bundle_to_carrier(bundle, carrier_copy)
        raise NotImplementedError(f'This has never been used before, check before removing the Error raise')

        # sequentially insert bundle requests
        try:
            while carrier_copy.unrouted_requests:
                request = carrier_copy.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution_copy, carrier_copy.id_, request)

            solution_copy_improved = self.tour_improvement.execute(instance, solution_copy, [carrier_id])
            carrier_copy_improved = solution_copy_improved.carriers[carrier_copy.id_]

            with_bundle = carrier_copy_improved.objective()

        except utility_module.errors.ConstraintViolationError:
            with_bundle = -float('inf')

        return with_bundle


class ClearAndReinsertAll(BiddingBehavior):
    """
    The profit for the carrier WITH the bundle added is calculated by inserting all requests (the ones that already
    belong to the carrier AND the bundle's requests) sequentially in their order of vertex index, i.e. in the order
    of request arrival to mimic the dynamic request arrival process within the acceptance phase. Afterwards, an
    improvement is executed using the defined metaheuristic.
    Note that this is only compatible if the same dynamic re-optimization is also applied after the reallocation, too.
    """

    def _value_with_bundle(self, instance: it.MDPDPTWInstance, pre_rs_solution: slt.CAHDSolution,
                           post_rs_solution: slt.CAHDSolution, bundle: Sequence[int], carrier_id: int):

        post_rs_solution_copy = deepcopy(post_rs_solution)
        post_rs_carrier_copy = post_rs_solution_copy.carriers[carrier_id]
        pre_rs_carrier = pre_rs_solution.carriers[carrier_id]
        self._add_bundle_to_carrier(bundle, post_rs_carrier_copy, pre_rs_carrier)

        # reset the temporary carrier's solution and start from scratch instead
        post_rs_solution_copy.clear_carrier_routes([post_rs_carrier_copy.id_])

        # sequentially insert requests (original + bundle) just as if it was the acceptance phase
        try:
            while post_rs_carrier_copy.unrouted_requests:
                request = post_rs_carrier_copy.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, post_rs_solution_copy, post_rs_carrier_copy.id_,
                                                             request)

            solution_copy_improved = self.tour_improvement.execute(instance, post_rs_solution_copy, [carrier_id])
            carrier_copy_improved = solution_copy_improved.carriers[post_rs_carrier_copy.id_]

            with_bundle = carrier_copy_improved.objective()

        except utility_module.errors.ConstraintViolationError:
            with_bundle = -float('inf')

        return with_bundle
