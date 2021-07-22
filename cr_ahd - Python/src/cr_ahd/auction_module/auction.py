import logging
from abc import ABC
from copy import deepcopy

from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)


class Auction(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPMetaHeuristic,
                 request_selection: rs.RequestSelectionBehavior,
                 bundle_generation: bg.BundlePoolGenerationBehavior,
                 bidding: bd.BiddingBehavior,
                 winner_determination: wd.WinnerDeterminationBehavior,
                 ):
        """
        Auction class can be called with various parameters to create different auction variations

        :param tour_construction: construction method
        :param tour_improvement: improvement metaheuristic
        :param request_selection: request selection method
        :param bundle_generation:
        :param bidding: bidding behavior, in bidding, the same construction & improvement methods must be used as
        are specified for this Auction to guarantee consistent, individual-rational auction results
        :param winner_determination:
        :param reopt_and_improve_after_request_selection: shall a reoptimization happen after the requests have been
        submitted? As long as I'm doing complete, dynamic reoptimization in bidding, this should be set to True
        """

        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.request_selection = request_selection
        self.bundle_generation = bundle_generation
        self.bidding = bidding
        self.winner_determination = winner_determination

        assert isinstance(self.bidding.tour_construction, type(self.tour_construction))
        assert isinstance(self.bidding.tour_improvement, type(self.tour_improvement))

    def execute_auction(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        # Request Selection
        auction_request_pool, original_bundling_labels = self.request_selection.execute(instance, solution)
        if auction_request_pool:
            logger.debug(f'requests {auction_request_pool} have been submitted to the auction pool')

            # optional reoptimization, not all auction variants do a reopt, the abstract method may be empty
            # TODO does this even make any sense if i do complete dynamic reopt in bidding anyways?!
            # if self.reopt_and_improve_after_request_selection:
            #     solution = self._reopt_and_improve(instance, solution)

            # Bundle Generation
            auction_bundle_pool = self.bundle_generation.execute_bundle_pool_generation(instance, solution, auction_request_pool,
                                                                                        original_bundling_labels)
            original_bundles = ut.indices_to_nested_lists(original_bundling_labels, auction_request_pool)
            original_bundles_indices = [auction_bundle_pool.index(x) for x in original_bundles]
            logger.debug(f'bundles {auction_bundle_pool} have been created')

            # Bidding
            logger.debug(f'Generating bids_matrix')
            bids_matrix = self.bidding.execute_bidding(instance, solution, auction_bundle_pool)
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_bundle_pool}')

            # Winner Determination
            winner_bundles, bundle_winners = self.winner_determination.execute(instance, solution,
                                                                               auction_request_pool,
                                                                               auction_bundle_pool, bids_matrix)
            winner_bundles_indices = [auction_bundle_pool.index(x) for x in winner_bundles]

            # bundle reallocation
            self.assign_bundles_to_winners(solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        # final routing
        # clear the solution and do a dynamic re-optimization + improvement
        solution.clear_carrier_routes()
        for carrier in range(len(solution.carriers)):
            while solution.carriers[carrier].unrouted_requests:
                request = solution.carriers[carrier].unrouted_requests[0]
                self.tour_construction.construct_dynamic(instance, solution, carrier)
        self.tour_improvement.execute(instance, solution)


        pass

    @staticmethod
    def assign_bundles_to_winners(solution, winner_bundles, bundle_winners):
        # assign the bundles to the corresponding winner
        for bundle, winner in zip(winner_bundles, bundle_winners):
            solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
            solution.carriers[winner].accepted_requests.extend(bundle)
        # must be sorted to obtain the acceptance phase's solutions also in the final routing
        for carrier_ in solution.carriers:
            carrier_.assigned_requests.sort()
            carrier_.accepted_requests.sort()
            carrier_.unrouted_requests.sort()
