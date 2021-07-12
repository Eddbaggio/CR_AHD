import logging
from abc import ABC, abstractmethod
from typing import Sequence

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh, tour_initialization as ini

logger = logging.getLogger(__name__)


class Auction(ABC):
    def __init__(self,
                 construction_method: cns.PDPParallelInsertionConstruction,
                 improvement_method: mh.PDPMetaHeuristic):
        self.construction_method = construction_method
        self.improvement_method = improvement_method

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        self._run_auction(instance, solution)
        solution.auction_mechanism = self.__class__.__name__
        pass

    def _run_auction(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        # Request Selection
        auction_pool_requests, original_bundles = self._request_selection(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

        if auction_pool_requests:
            logger.debug(f'requests {auction_pool_requests} have been submitted to the auction pool')

            # improve the "reduced" routes
            self.improvement_method.execute(instance, solution)

            # Bundle Generation
            # TODO maybe bundles should be a list of bundle indices rather than a list of lists of request indices?
            auction_pool_bundles = self._bundle_generation(instance, solution, auction_pool_requests, original_bundles)
            logger.debug(f'bundles {auction_pool_bundles} have been created')

            # Bidding
            logger.debug(f'Generating bids_matrix')
            bids_matrix = self._bid_generation(instance, solution, auction_pool_bundles)

            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_pool_bundles}')

            # Winner Determination
            winner_bundles, bundle_winners = self._winner_determination(instance, solution, auction_pool_requests,
                                                                        auction_pool_bundles, bids_matrix)
            # assign the bundles to the corresponding winner
            for bundle, winner in zip(winner_bundles, bundle_winners):
                solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
                solution.carriers[winner].accepted_requests.extend(bundle)

            # must be sorted to obtain the acceptance phase's solutions also in the final routing
            for carrier_ in solution.carriers:
                carrier_.assigned_requests.sort()
                carrier_.accepted_requests.sort()
                carrier_.unrouted_requests.sort()
        else:
            logger.warning(f'No requests have been submitted!')
        pass

    @abstractmethod
    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        pass

    @abstractmethod
    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool, original_bundles):
        pass

    @abstractmethod
    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        pass

    @abstractmethod
    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        pass


class AuctionD(Auction):
    """
    Request Selection Behavior: Cluster \n
    Bundle Generation Behavior: Genetic Algorithm by Gansterer & Hartl \n
    Bidding Behavior: Profit \n
    Winner Determination Behavior: Gurobi - Set Packing Problem
    """

    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        return rs.Cluster().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool,
                           original_bundles):
        return bg.GeneticAlgorithm().execute(instance, solution, auction_pool, original_bundles)

    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        return bd.DynamicReOpt(construction_method=self.construction_method,
                               improvement_method=self.improvement_method).execute(instance, solution, bundles)

    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        return wd.MaxBidGurobiCAP1().execute(instance, solution, auction_pool, bundles, bids_matrix)
