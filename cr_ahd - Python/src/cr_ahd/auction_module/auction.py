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
        self._final_routing(instance, solution)
        solution.auction_mechanism = self.__class__.__name__
        pass

    def _run_auction(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        # Request Selection
        auction_pool_requests, original_bundles_indices = self._request_selection(instance, solution,
                                                                                  ut.NUM_REQUESTS_TO_SUBMIT)

        if auction_pool_requests:
            logger.debug(f'requests {auction_pool_requests} have been submitted to the auction pool')
            # optional, not all auction variants to a reopt, the abstract method may be empty
            self.reopt_and_improve_after_request_selection(instance, solution)

            # Bundle Generation
            # TODO maybe bundles should be a list of bundle indices rather than a list of lists of request indices?
            auction_pool_bundles = self._bundle_generation(instance, solution, auction_pool_requests,
                                                           original_bundles_indices)

            original_bundles = ut.indices_to_nested_lists(original_bundles_indices, auction_pool_requests)
            original_bundles_indices = [auction_pool_bundles.index(x) for x in original_bundles]

            logger.debug(f'bundles {auction_pool_bundles} have been created')

            # Bidding
            logger.debug(f'Generating bids_matrix')
            bids_matrix = self._bid_generation(instance, solution, auction_pool_bundles)

            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_pool_bundles}')

            # Winner Determination
            winner_bundles, bundle_winners = self._winner_determination(instance, solution, auction_pool_requests,
                                                                        auction_pool_bundles, bids_matrix)
            winner_bundles_indices = [auction_pool_bundles.index(x) for x in winner_bundles]

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
    def reopt_and_improve_after_request_selection(self, instance, solution):
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

    @abstractmethod
    def _final_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        pass


class AuctionDynamicReOptAndImprove(Auction):
    """
    Request Selection Behavior: Cluster \n
    Bundle Generation Behavior: Genetic Algorithm by Gansterer & Hartl \n
    Bidding Behavior: Profit \n
    Winner Determination Behavior: Gurobi - Set Packing Problem
    """

    def reopt_and_improve_after_request_selection(self, instance, solution):
        # clear the solution and do a dynamic re-optimization to get proper value_without_bundle values in bidding
        solution.clear_carrier_routes()
        for carrier in range(len(solution.carriers)):
            self.construction_method.construct_dynamic(instance, solution, carrier)
        self.improvement_method.execute(instance, solution)
        pass

    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        return rs.Cluster().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool,
                           original_bundles):
        return bg.GeneticAlgorithm().execute(instance, solution, auction_pool, original_bundles)

    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        # select a dynamic re-optimization policy for bidding
        return bd.DynamicReOptAndImprove(construction_method=self.construction_method,
                                         improvement_method=self.improvement_method).execute(instance, solution,
                                                                                             bundles)

    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        return wd.MaxBidGurobiCAP1().execute(instance, solution, auction_pool, bundles, bids_matrix)

    def _final_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        # do a final dynamic re-optimization from scratch
        solution.clear_carrier_routes()
        for carrier in range(solution.num_carriers()):
            while solution.carriers[carrier].unrouted_requests:
                self.construction_method.construct_dynamic(instance, solution, carrier)
        self.improvement_method.execute(instance, solution)


class AuctionStaticInsertion(Auction):
    raise NotImplementedError
    # the idea was good, however, even if a carrier re-inserts his original requests in a static fashion, he might end
    # up creating a solution that's worse than what he had obtained via dynamic insertion before the Request Selection

    def reopt_and_improve_after_request_selection(self, instance, solution):
        # don't do any reoptimization or improvements after request selection
        pass

    def _request_selection(self, instance: it.PDPInstance, solution: slt.CAHDSolution, num_request_to_submit: int):
        return rs.Cluster().execute(instance, solution, ut.NUM_REQUESTS_TO_SUBMIT)

    def _bundle_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool, original_bundles):
        return bg.GeneticAlgorithm().execute(instance, solution, auction_pool, original_bundles)

    def _bid_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundles):
        # choose a static insertion policy for bidding that does not do complete reoptimization
        return bd.StaticInsertion(self.construction_method, self.improvement_method).execute(instance, solution,
                                                                                             bundles)

    def _winner_determination(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence[int],
                              bundles: Sequence[Sequence[int]], bids_matrix: Sequence[Sequence[float]]):
        return wd.MaxBidGurobiCAP1().execute(instance, solution, auction_pool, bundles, bids_matrix)

    def _final_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        # do the final routing just the same way the bids were generated, but we add an improvement phase to the end
        for carrier in range(len(solution.carriers)):
            while solution.carriers[carrier].unrouted_requests:
                self.construction_method.construct_static(instance, solution, carrier)
        self.improvement_method.execute(instance, solution)
