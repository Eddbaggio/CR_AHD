import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_initialization as ini, tour_improvement as imp

logger = logging.getLogger(__name__)


class Auction(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        self._run_auction(instance, solution)
        solution.auction_mechanism = self.__class__.__name__
        pass

    def _run_auction(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        # Request Selection
        auction_pool_requests, original_bundles = self._request_selection(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

        if auction_pool_requests:
            # add the requests (that have not been submitted for auction) to the tours
            logger.debug(f'requests {auction_pool_requests} have been submitted to the auction pool')
            logger.debug(f'routing non-submitted requests {[c.unrouted_requests for c in solution.carriers]}')
            # todo this must happen in the same dynamic/incremental way in which the original routing happened. However,
            #  even then, it is not guaranteed that a solution will be found!
            self._route_unsubmitted(instance, solution)

            # Bundle Generation
            # todo bundles should be a list of bundle indices rather than a list of lists of request indices
            auction_pool_bundles = self._bundle_generation(instance, solution, auction_pool_requests, original_bundles)
            logger.debug(f'bundles {auction_pool_bundles} have been created')

            # Bidding
            logger.debug(f'Generating bids_matrix')
            # todo this must happen in the same dynamic/incremental way in which the original routing happened. Here,
            #  it is guaranteed that at least the original solution will be found for the original bundles
            bids_matrix = self._bid_generation(instance, solution, auction_pool_bundles)

            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_pool_bundles}')

            # Winner Determination
            winner_bundles, bundle_winners = self._winner_determination(instance, solution, auction_pool_requests,
                                                                        auction_pool_bundles,
                                                                        bids_matrix)
            # logger.debug(f'reassigning bundles {winner_bundles} to carriers {bundle_winners}')

            for bundle, winner in zip(winner_bundles, bundle_winners):
                solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
        else:
            logger.warning(f'No requests have been submitted!')
        pass

    @abstractmethod
    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        """

        :param instance:
        :param solution:
        :param num_request_to_submit:
        :return:
        """
        pass

    @abstractmethod
    def _route_unsubmitted(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        pass

    @abstractmethod
    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool,
                           original_bundles):
        pass

    @abstractmethod
    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        pass

    @abstractmethod
    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
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



class AuctionC(Auction):
    """
    Request Selection Behavior: Lowest Profit \n
    Bundle Generation Behavior: Genetic Algorithm by Gansterer & Hartl \n
    Bidding Behavior: Profit \n
    Winner Determination Behavior: Gurobi - Set Packing Problem
    """

    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        return rs.LowestProfit().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _route_unsubmitted(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        cns.CheapestPDPInsertion().construct(instance, solution)

    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool,
                           original_bundles):
        return bg.GeneticAlgorithm().execute(instance, solution, auction_pool, original_bundles)

    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        return bd.Profit().execute(instance, solution, bundles)

    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        return wd.MaxBidGurobiCAP1().execute(instance, solution, auction_pool, bundles, bids_matrix)
'''


class AuctionD(Auction):
    """
    Request Selection Behavior: Cluster \n
    Bundle Generation Behavior: Genetic Algorithm by Gansterer & Hartl \n
    Bidding Behavior: Profit \n
    Winner Determination Behavior: Gurobi - Set Packing Problem
    """

    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        return rs.Cluster().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _route_unsubmitted(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        happening in the same dynamic/incremental way of the acceptance phase
        """
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # build tours with the assigned and tw-managed requests
            cns.CheapestPDPInsertion().construct(instance, solution)
            imp.PDPMoveBestImpr().improve_global_solution(instance, solution)

    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool,
                           original_bundles):
        return bg.GeneticAlgorithm().execute(instance, solution, auction_pool, original_bundles)

    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        return bd.Profit().execute(instance, solution, bundles)

    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        return wd.MaxBidGurobiCAP1().execute(instance, solution, auction_pool, bundles, bids_matrix)
