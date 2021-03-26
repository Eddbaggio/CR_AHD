from abc import ABC, abstractmethod
from typing import Dict, Tuple

from src.cr_ahd.core_module.vertex import Vertex
from src.cr_ahd.solving_module.tour_construction import I1Insertion
from src.cr_ahd.solving_module.tour_initialization import EarliestDueDate
from src.cr_ahd.utility_module.utils import InsertionError


class BiddingBehavior(ABC):
    def execute(self, bundle_set: Dict[int, Tuple[Vertex]], carriers):
        """

        :param bundle_set: dictionary of bundles where the key is simply a bundle index
        :param carriers:
        :return: dict of bids per bundle per carrier: {bundleA: {carrier1: bid, carrier2: bid}, bundleB: {carrier1: bid,
        carrier2: bid} Dict[Tuple[Vertex], Dict[Carrier, float]]
        """
        bundle_bids = dict()
        for _, bundle in bundle_set.items():
            carrier_bundle_bids = dict()
            for carrier in carriers:
                carrier_bundle_bids[carrier] = self._generate_bid(bundle, carrier)
            bundle_bids[bundle] = carrier_bundle_bids
        # solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    @abstractmethod
    def _generate_bid(self, bundle, carrier):
        pass


class I1MarginalCostBidding(BiddingBehavior):
    """determine the bids as the marginal cost of inserting a bundle set. Marginal profit is difference in route cost
    with and without the set where insertion is done with I1 Insertion scheme"""

    def _generate_bid(self, bundle, carrier):

        without_bundle = carrier.sum_travel_durations()
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

        with_bundle = carrier.sum_travel_durations()
        carrier.retract_requests_and_update_routes(bundle)
        carrier.retract_requests_and_update_routes(prior_unrouted)  # need to unroute the previously unrouted again
        carrier.assign_requests(prior_unrouted)  # and then reassign them (missing a function that only unroutes them)
        marginal_cost = with_bundle - without_bundle
        return marginal_cost
