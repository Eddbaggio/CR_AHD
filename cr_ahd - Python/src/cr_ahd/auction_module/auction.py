from abc import ABC, abstractmethod

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
import logging

logger = logging.getLogger(__name__)


class Auction(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        self._run_auction(instance, solution)
        solution.auction_mechanism = self.__class__.__name__
        pass

    @abstractmethod
    def _run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


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


class AuctionC(Auction):
    """
    Request Selection Behavior: Highest Insertion Cost Distance
    Bundle Generation Behavior: K-Means
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        submitted_requests = rs.HighestInsertionCostDistance().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)
        logger.debug(f'requests {submitted_requests} have been submitted to the auction pool')
        if submitted_requests:
            bundles = bg.KMeansBundles().execute(instance, solution, submitted_requests)
            logger.debug(f'bundles {bundles} have been created')
            logger.debug(f'Generating bids')
            bids = bd.CheapestInsertionDistanceIncrease().execute(instance, solution, bundles)
            logger.debug(f'Bids {bids} have been created for bundles {bundles}')
            bundle_winners = wd.LowestBid().execute(instance, solution, bundles, bids)
            logger.debug(f'reassigning bundles {bundles} to carriers {bundle_winners}')
            for bundle, winner in zip(bundles, bundle_winners):
                solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))

