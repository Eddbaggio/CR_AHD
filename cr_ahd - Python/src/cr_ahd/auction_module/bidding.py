from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.routing_module.tour_initialization import EarliestDueDate
from src.cr_ahd.utility_module.utils import InsertionError, ConstraintViolationError
from src.cr_ahd.core_module import instance as it, solution as slt
import logging

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
        cs_copy = deepcopy(solution.carrier_solutions[carrier])  # create a temporary copy
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


class I1TravelDistanceIncrease(BiddingBehavior):
    """determine the bids as the distance cost of inserting a bundle set. Marginal profit is difference in route cost
    with and without the set where insertion is done with I1 Insertion scheme"""

    def _generate_bid(self, bundle, carrier):

        without_bundle = carrier.sum_travel_distance()
        prior_unrouted = carrier.unrouted_requests
        carrier.assign_requests(bundle)

        # TODO duplicated code fragment, is there a nicer way to do this? do error codes instead of exception handling?
        c = True
        while c:
            try:
                I1Insertion().solve_carrier(carrier)
                c = False
            except InsertionError:
                EarliestDueDate().initialize_carrier(carrier)

        with_bundle = carrier.sum_travel_distance()
        carrier.retract_requests_and_update_routes(bundle)
        carrier.retract_requests_and_update_routes(prior_unrouted)  # need to unroute the previously unrouted again
        carrier.assign_requests(prior_unrouted)  # and then reassign them (missing a function that only unroutes them)
        distance_increase = with_bundle - without_bundle
        return distance_increase


class I1TravelDurationIncrease(BiddingBehavior):
    """determine the bids as the distance cost of inserting a bundle set. Marginal profit is difference in route cost
    with and without the set where insertion is done with I1 Insertion scheme"""

    def _generate_bid(self, bundle, carrier):

        without_bundle = carrier.sum_travel_duration()
        prior_unrouted = carrier.unrouted_requests
        carrier.assign_requests(bundle)

        # TODO duplicated code fragment, is there a nicer way to do this? do error codes instead of exception handling?
        c = True
        while c:
            try:
                I1Insertion().solve_carrier(carrier)
                c = False
            except InsertionError:
                EarliestDueDate().initialize_carrier(carrier)

        with_bundle = carrier.sum_travel_duration()
        carrier.retract_requests_and_update_routes(bundle)
        carrier.retract_requests_and_update_routes(prior_unrouted)  # need to unroute the previously unrouted again
        carrier.assign_requests(prior_unrouted)  # and then reassign them (missing a function that only unroutes them)
        duration_increase = with_bundle - without_bundle
        return duration_increase
