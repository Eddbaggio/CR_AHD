from abc import ABC, abstractmethod

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
import logging

logger = logging.getLogger(__name__)


class Auction(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        self._run_auction(instance, solution)
        solution.auction_mechanism = self.__class__.__name__
        pass

    def _run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        submitted_requests = self._request_selection(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)
        if submitted_requests:
            logger.debug(f'requests {submitted_requests} have been submitted to the auction pool')
            logger.debug(f'routing non-submitted requests {[c.unrouted_requests for c in solution.carriers]}')
            self._route_unsubmitted(instance, solution)
            bundles = self._bundle_generation(instance, solution, submitted_requests)
            logger.debug(f'bundles {bundles} have been created')
            logger.debug(f'Generating bids')
            bids = self._bid_generation(instance, solution, bundles)
            logger.debug(f'Bids {bids} have been created for bundles {bundles}')
            winner_bundles, bundle_winners = self._winner_determination(instance, solution, bundles, bids)
            logger.debug(f'reassigning bundles {winner_bundles} to carriers {bundle_winners} for bids')
            for bundle, winner in zip(winner_bundles, bundle_winners):
                solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
        else:
            logger.warning(f'No requests have been submitted!')
        pass

    @abstractmethod
    def _request_selection(self, instance, solution, num_request_to_submit):
        """

        :param instance:
        :param solution:
        :param num_request_to_submit:
        :return:
        """
        pass

    @abstractmethod
    def _route_unsubmitted(self, instance, solution):
        pass

    @abstractmethod
    def _bundle_generation(self, instance, solution, submitted_requests):
        pass

    @abstractmethod
    def _bid_generation(self, instance, solution, bundles):
        pass

    @abstractmethod
    def _winner_determination(self, instance, solution, bundles, bids):
        pass


'''
class AuctionA(Auction):
    """
    Request Selection Behavior: Highest Insertion Cost Distance
    Bundle Generation Behavior: Random Partition
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        submitted_requests = rs.HighestInsertionCostDistance().execute(instance, ut.NUM_REQUESTS_TO_SUBMIT)
        bundle_set = bg.RandomPartition(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)


class AuctionB(Auction):
    """
    Request Selection Behavior: Cluster (Gansterer & Hartl 2016)
    Bundle Generation Behavior: Random Partition
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        submitted_requests = rs.Cluster().execute(instance, ut.NUM_REQUESTS_TO_SUBMIT)
        bundle_set = bg.RandomPartition(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)
'''


class AuctionC(Auction):
    """
    Request Selection Behavior: Lowest Profit \n
    Bundle Generation Behavior: Power Set - All Bundles \n
    Bidding Behavior: Profit \n
    Winner Determination Behavior: Gurobi - Set Packing Problem
    """

    def _request_selection(self, instance, solution, num_request_to_submit):
        return rs.LowestProfit().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _route_unsubmitted(self, instance, solution):
        cns.CheapestPDPInsertion().construct(instance, solution)
        pass

    def _bundle_generation(self, instance, solution, submitted_requests):
        return bg.ProxyTest().execute(instance, solution, submitted_requests)

    def _bid_generation(self, instance, solution, bundles):
        return bd.Profit().execute(instance, solution, bundles)

    def _winner_determination(self, instance, solution, bundles, bids):
        return wd.MaxBidGurobi().execute(instance, solution, bundles, bids)


